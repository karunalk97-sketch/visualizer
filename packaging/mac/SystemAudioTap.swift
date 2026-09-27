// SystemAudioTap -- lets the visualizer hear everything the Mac is playing, with no
// virtual audio driver to install. It uses a Core Audio process tap (macOS 14.2+):
// macOS asks once for permission to record system audio, then the mixed output of
// every app streams through here.
//
// stdout: a 12-byte header ("VTAP", UInt32 version = 1, UInt32 sample rate, little
//         endian), sent with the first audio, then mono Float32 samples.
// stderr: "hint: no-permission" when apps are playing but only silence arrives
//         (permission denied), "error: ..." when the tap can't be created.
// It exits when stdin closes (the app quit) or when the default output device
// changes (the app then starts a fresh copy on the new device).
//
//   swiftc -O -target arm64-apple-macos14.2 SystemAudioTap.swift -o SystemAudioTap

import CoreAudio
import Foundation

func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func fail(_ s: String, _ code: Int32) -> Never {
    note("error: " + s)
    exit(code)
}

func writeAll(_ p: UnsafeRawPointer, _ n: Int) {
    var off = 0
    while off < n {
        let w = write(1, p + off, n - off)
        if w < 0 && errno == EINTR { continue }
        if w <= 0 { exit(0) }                                // the app is gone
        off += w
    }
}

func address(_ sel: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

func readValue<T>(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ initial: T) -> T? {
    var addr = address(sel)
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    let status = withUnsafeMutableBytes(of: &value) { AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, $0.baseAddress!) }
    return status == noErr ? value : nil
}

func readString(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector) -> String? {
    var addr = address(sel)
    var ref: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, &ref) == noErr, let r = ref else { return nil }
    return r.takeRetainedValue() as String
}

/// Shared between the audio queue and the main queue.
final class Flags {
    private let lock = NSLock()
    private var heard = false
    var headerSent = false                                   // only touched on the audio queue
    func markHeard() { lock.lock(); heard = true; lock.unlock() }
    func takeHeard() -> Bool { lock.lock(); defer { heard = false; lock.unlock() }; return heard }
}

let system = AudioObjectID(kAudioObjectSystemObject)

/// Is any other app sending audio to an output right now?
func somethingElsePlaying() -> Bool {
    var addr = address(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr, size > 0 else { return false }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return false }
    let mine: Set<pid_t> = [getpid(), getppid()]
    return ids.contains { id in
        (readValue(id, kAudioProcessPropertyIsRunningOutput, UInt32(0)) ?? 0) != 0
            && !mine.contains(readValue(id, kAudioProcessPropertyPID, pid_t(0)) ?? 0)
    }
}

signal(SIGPIPE, SIG_IGN)
if CommandLine.arguments.contains("--check") {
    print("ok")
    exit(0)
}

// The app closes stdin (or dies): stop. Straight from this thread, because the main
// thread may be waiting on the permission prompt; the tap and the capture device are
// private to this process, so they go away with it.
Thread.detachNewThread {
    while !FileHandle.standardInput.availableData.isEmpty {}
    exit(0)
}

guard let output = readValue(system, kAudioHardwarePropertyDefaultOutputDevice, AudioObjectID(kAudioObjectUnknown)),
      output != AudioObjectID(kAudioObjectUnknown),
      let outputUID = readString(output, kAudioDevicePropertyDeviceUID)
else { fail("no output device", 2) }
note("status: output \(outputUID)")

// A private tap on everything the Mac plays; the sound still reaches the speakers.
let tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
tapDescription.name = "Audio Visualizer"
tapDescription.isPrivate = true
tapDescription.muteBehavior = .unmuted
var tap = AudioObjectID(kAudioObjectUnknown)
var status = AudioHardwareCreateProcessTap(tapDescription, &tap)
guard status == noErr else { fail("couldn't create the system audio tap (\(status))", 4) }
note("status: tap created")

guard let format = readValue(tap, kAudioTapPropertyFormat, AudioStreamBasicDescription()),
      format.mFormatID == kAudioFormatLinearPCM, format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
      format.mBitsPerChannel == 32
else { fail("unexpected tap format", 4) }

// Taps are read through an aggregate device clocked by the current output device.
let aggregateDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey: "Audio Visualizer Tap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
    kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                       kAudioSubTapUIDKey: tapDescription.uuid.uuidString]],
]
var aggregate = AudioObjectID(kAudioObjectUnknown)
status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregate)
guard status == noErr else {
    AudioHardwareDestroyProcessTap(tap)
    fail("couldn't create the capture device (\(status))", 4)
}

note("status: device created")

let flags = Flags()
let maxFrames = 1 << 15
let mono = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames)
var header = [UInt8]("VTAP".utf8)
for v in [UInt32(1), UInt32(format.mSampleRate)] { withUnsafeBytes(of: v.littleEndian) { header += $0 } }

var proc: AudioDeviceIOProcID?
status = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregate, DispatchQueue(label: "tap", qos: .userInteractive)) {
    _, input, _, _, _ in
    let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
    guard let first = buffers.first, first.mNumberChannels > 0 else { return }
    let frames = min(Int(first.mDataByteSize) / 4 / Int(first.mNumberChannels), maxFrames)
    guard frames > 0 else { return }
    mono.update(repeating: 0, count: frames)
    var channels = 0
    for b in buffers {                                       // interleaved or one buffer per channel
        let ch = Int(b.mNumberChannels)
        guard ch > 0, let data = b.mData?.assumingMemoryBound(to: Float.self),
              Int(b.mDataByteSize) / 4 >= frames * ch else { continue }
        for i in 0..<frames {
            var s: Float = 0
            for c in 0..<ch { s += data[i * ch + c] }
            mono[i] += s
        }
        channels += ch
    }
    guard channels > 0 else { return }
    let scale = 1 / Float(channels)
    var heard = false
    for i in 0..<frames {
        mono[i] *= scale
        if mono[i] != 0 { heard = true }
    }
    if heard { flags.markHeard() }
    if !flags.headerSent {
        header.withUnsafeBytes { writeAll($0.baseAddress!, $0.count) }
        flags.headerSent = true
    }
    writeAll(mono, frames * 4)
}
guard status == noErr, let procID = proc else { fail("couldn't read the capture device (\(status))", 4) }

let cleanup = {
    AudioDeviceStop(aggregate, procID)
    AudioDeviceDestroyIOProcID(aggregate, procID)
    AudioHardwareDestroyAggregateDevice(aggregate)
    AudioHardwareDestroyProcessTap(tap)
}

// The first time, macOS asks for permission here and this waits for the answer.
status = AudioDeviceStart(aggregate, procID)
guard status == noErr else {
    cleanup()
    fail("couldn't start capturing (\(status))", 4)
}
note("status: capturing")

// A new default output (headphones plugged in...): quit, the app starts a fresh copy.
var defaultOutput = address(kAudioHardwarePropertyDefaultOutputDevice)
AudioObjectAddPropertyListenerBlock(system, &defaultOutput, DispatchQueue.main) { _, _ in
    cleanup()
    exit(0)
}

// Apps are playing but only exact silence arrives for a few seconds: macOS is
// withholding the audio, i.e. the permission was denied.
var quietWhilePlaying = 0
let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.main)
timer.schedule(deadline: .now() + 1, repeating: 1)
timer.setEventHandler {
    if flags.takeHeard() || !somethingElsePlaying() {
        quietWhilePlaying = 0
        return
    }
    quietWhilePlaying += 1
    if quietWhilePlaying == 4 { note("hint: no-permission") }
}
timer.resume()

dispatchMain()
