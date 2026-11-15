//
//  VideoExportService.swift
//  MobileSana
//
//  Created by Claude on 2025/6/9.
//

import AVFoundation
import CoreGraphics
import Foundation
import Photos
import UIKit

class VideoExportService {
  enum ExportError: LocalizedError {
    case noFrames
    case exportFailed(String)
    case permissionDenied
    case invalidFrameData

    var errorDescription: String? {
      switch self {
      case .noFrames:
        return "No video frames available to export"
      case .exportFailed(let message):
        return "Export failed: \(message)"
      case .permissionDenied:
        return "Photo library access permission denied"
      case .invalidFrameData:
        return "Invalid frame data"
      }
    }
  }

  static let shared = VideoExportService()

  private init() {}

  func exportVideo(
    frames: [CGImage],
    fps: Double = 10.0,
    progressHandler: @escaping (Float) -> Void,
    completionHandler: @escaping (Result<URL, ExportError>) -> Void
  ) {
    guard !frames.isEmpty else {
      completionHandler(.failure(.noFrames))
      return
    }

    Task {
      do {
        let videoURL = try await createVideoFromFrames(
          frames: frames,
          fps: fps,
          progressHandler: progressHandler
        )
        await MainActor.run {
          completionHandler(.success(videoURL))
        }
      } catch {
        await MainActor.run {
          if let exportError = error as? ExportError {
            completionHandler(.failure(exportError))
          } else {
            completionHandler(.failure(.exportFailed(error.localizedDescription)))
          }
        }
      }
    }
  }

  func saveVideoToPhotoLibrary(
    frames: [CGImage],
    fps: Double = 10.0,
    progressHandler: @escaping (Float) -> Void,
    completionHandler: @escaping (Result<Void, ExportError>) -> Void
  ) {
    // First check permission
    let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)

    if status == .denied || status == .restricted {
      completionHandler(.failure(.permissionDenied))
      return
    }

    if status == .notDetermined {
      PHPhotoLibrary.requestAuthorization(for: .addOnly) { newStatus in
        DispatchQueue.main.async {
          if newStatus == .authorized || newStatus == .limited {
            self.performVideoSave(
              frames: frames,
              fps: fps,
              progressHandler: progressHandler,
              completionHandler: completionHandler
            )
          } else {
            completionHandler(.failure(.permissionDenied))
          }
        }
      }
      return
    }

    performVideoSave(
      frames: frames,
      fps: fps,
      progressHandler: progressHandler,
      completionHandler: completionHandler
    )
  }

  private func performVideoSave(
    frames: [CGImage],
    fps: Double,
    progressHandler: @escaping (Float) -> Void,
    completionHandler: @escaping (Result<Void, ExportError>) -> Void
  ) {
    Task {
      do {
        let videoURL = try await createVideoFromFrames(
          frames: frames,
          fps: fps,
          progressHandler: progressHandler
        )

        try await saveToPhotoLibrary(videoURL: videoURL)

        // Clean up temporary file
        try? FileManager.default.removeItem(at: videoURL)

        await MainActor.run {
          completionHandler(.success(()))
        }
      } catch {
        await MainActor.run {
          if let exportError = error as? ExportError {
            completionHandler(.failure(exportError))
          } else {
            completionHandler(.failure(.exportFailed(error.localizedDescription)))
          }
        }
      }
    }
  }

  private func createVideoFromFrames(
    frames: [CGImage],
    fps: Double,
    progressHandler: @escaping (Float) -> Void
  ) async throws -> URL {
    let videoSize = CGSize(width: frames[0].width, height: frames[0].height)
    let documentsPath = FileManager.default.temporaryDirectory
    let videoURL = documentsPath.appendingPathComponent("exported_video_\(UUID().uuidString).mp4")

    // Remove any existing file
    try? FileManager.default.removeItem(at: videoURL)

    let writer = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)

    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: videoSize.width,
      AVVideoHeightKey: videoSize.height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: 2_000_000,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264BaselineAutoLevel,
      ],
    ]

    let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: writerInput,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
        kCVPixelBufferWidthKey as String: videoSize.width,
        kCVPixelBufferHeightKey as String: videoSize.height,
      ]
    )

    writer.add(writerInput)

    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))

    return try await withCheckedThrowingContinuation { continuation in
      var frameIndex = 0
      var presentationTime = CMTime.zero

      writerInput.requestMediaDataWhenReady(on: DispatchQueue.global(qos: .userInitiated)) {
        while frameIndex < frames.count {
          // Check if we can write more data
          guard writerInput.isReadyForMoreMediaData else {
            // Wait a bit and try again in the next iteration
            break
          }

          guard
            let pixelBuffer = self.createPixelBuffer(
              from: frames[frameIndex],
              pixelBufferPool: adaptor.pixelBufferPool
            )
          else {
            writer.finishWriting {
              continuation.resume(throwing: ExportError.invalidFrameData)
            }
            return
          }

          adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
          presentationTime = CMTimeAdd(presentationTime, frameDuration)
          frameIndex += 1

          let progress = Float(frameIndex) / Float(frames.count)
          DispatchQueue.main.async {
            progressHandler(progress)
          }
        }

        // Check if all frames are processed
        if frameIndex >= frames.count {
          writerInput.markAsFinished()
          writer.finishWriting {
            if writer.status == .completed {
              continuation.resume(returning: videoURL)
            } else {
              let error = writer.error?.localizedDescription ?? "Unknown error"
              continuation.resume(throwing: ExportError.exportFailed(error))
            }
          }
        }
      }
    }
  }

  private func createPixelBuffer(from cgImage: CGImage, pixelBufferPool: CVPixelBufferPool?)
    -> CVPixelBuffer?
  {
    var pixelBuffer: CVPixelBuffer?

    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault, pixelBufferPool!, &pixelBuffer)

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    let context = CGContext(
      data: CVPixelBufferGetBaseAddress(buffer),
      width: CVPixelBufferGetWidth(buffer),
      height: CVPixelBufferGetHeight(buffer),
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
    )

    context?.draw(
      cgImage,
      in: CGRect(
        x: 0, y: 0, width: CGFloat(CVPixelBufferGetWidth(buffer)),
        height: CGFloat(CVPixelBufferGetHeight(buffer))))

    return buffer
  }

  private func saveToPhotoLibrary(videoURL: URL) async throws {
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
      PHPhotoLibrary.shared().performChanges({
        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
      }) { success, error in
        if success {
          continuation.resume()
        } else {
          continuation.resume(
            throwing: error ?? ExportError.exportFailed("Failed to save to photo library"))
        }
      }
    }
  }
}
