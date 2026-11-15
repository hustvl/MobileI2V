//
//  SanaVideoContentView.swift
//  MobileSana
//
//  Created by Silver Yu on 2025/4/8.
//

import CoreGraphics
import CoreML
import Foundation
import PhotosUI
import SwiftUI

struct VideoContentView: View {
  @State private var videoFrames: [CGImage] = []
  @State private var isLoading = false
  @State private var errorMessage: String?
  @State private var currentFrameIndex = 0
  @State private var fps: Double = 10.0
  @State private var flowScore: Double = 0.6  // Default flow score
  @State private var seed: UInt64 = 42  // Default seed for reproducible experiments

  // 图片选择器状态
  @State private var imageSelection: PhotosPickerItem?
  @State private var selectedImage: UIImage?

  // Timer for auto-play
  @State private var timer: Timer?
  @State private var isPlaying = false

  // Video export states
  @State private var isExporting = false
  @State private var exportProgress: Float = 0.0
  @State private var exportMessage: String?
  @State private var showingExportSuccess = false

  var body: some View {
    NavigationView {
      ScrollView {
        VStack {
          // ---------------- START: Image Selection Area ----------------
          VStack {
            // 图片选择区域
            VStack {
              PhotosPicker(selection: $imageSelection, matching: .images) {
                VStack {
                  if let selectedImage = selectedImage {
                    Image(uiImage: selectedImage)
                      .resizable()
                      .scaledToFit()
                      .frame(maxHeight: 300)
                      .cornerRadius(8)
                      .padding(.bottom)  // Add some padding below the image
                  } else {
                    VStack {
                      Image(systemName: "photo.badge.plus.fill")  // More indicative icon
                        .font(.system(size: 50))
                        .foregroundColor(.gray)
                        .padding(.bottom, 8)
                      Text("Tap to select first frame image")
                        .font(.headline)
                        .foregroundColor(.gray)
                    }
                    .frame(minHeight: 150, idealHeight: 200)
                    .frame(maxWidth: .infinity)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.1)))
                    .padding(.bottom)  // Add some padding below the box
                  }
                }
              }
              .onChange(of: imageSelection) { newItem in
                Task {
                  if let data = try? await newItem?.loadTransferable(type: Data.self),
                    let uiImage = UIImage(data: data)
                  {
                    selectedImage = uiImage
                    errorMessage = nil  // Clear previous errors on new image selection
                    videoFrames = []  // Reset video frames when new image is selected
                    currentFrameIndex = 0  // Reset frame index
                    isPlaying = false  // Stop playback
                    timer?.invalidate()  // Stop timer
                    timer = nil
                  }
                }
              }
              // The distilled checkpoint always runs the original two-step schedule.
              Text("Inference Steps: 2 (distilled)")
                .padding(.horizontal)  // Add some horizontal padding
                .padding(.bottom, 5)  // Add some padding below the stepper

              if VideoPipeline.modelNames.contains(where: {
                (try? VideoPipeline.modelURL($0)) == nil
              }) {
                Text(
                  "Model files are not bundled. Add Transformer, VAEEncoder and VAEDecoder packages to the app's Models folder and rebuild."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("missingModelsNotice")
              }

              // Add Stepper for flowScore
              HStack {
                Text("Flow Score: \(String(format: "%.1f", flowScore))")
                Stepper("", value: $flowScore, in: 0.0...5.0, step: 0.2)
              }
              .padding(.horizontal)  // Add some horizontal padding
              .padding(.bottom, 5)  // Add some padding below the stepper

              // Add TextField for seed
              HStack {
                Text("Seed: ")
                TextField("Enter seed", value: $seed, format: .number)
                  .textFieldStyle(RoundedBorderTextFieldStyle())
                  .keyboardType(.numberPad)
              }
              .padding(.horizontal)  // Add some horizontal padding
              .padding(.bottom, 5)  // Add some padding below the field

              Button {
                Task { await generateVideo() }
              } label: {
                HStack {
                  Image(systemName: "play.rectangle.fill")  // Icon added here
                  Text("Process Video")
                }
              }
              .padding()
              .background(selectedImage != nil ? Color.blue : Color.gray)
              .foregroundColor(.white)
              .cornerRadius(8)
              .disabled(selectedImage == nil || isLoading)
            }
            .disabled(isLoading || isExporting)
          }
          // ---------------- END: Image Selection Area ----------------

          // ---------------- START: Conditional Display (Loading, Error, Video Player) ----------------
          if isLoading {
            ProgressView("Processing...")
              .padding()
          } else if let error = errorMessage {
            VStack {
              Text("Error")
                .font(.headline)
                .foregroundColor(.red)
              Text(error)
                .foregroundColor(.red)
                .multilineTextAlignment(.center)
                .padding()
              Button("Retry") {
                Task { await generateVideo() }
              }
              .padding()
              .background(Color.blue)
              .foregroundColor(.white)
              .cornerRadius(8)
            }
          } else if !videoFrames.isEmpty {
            // Video player
            VStack {
              // Display current frame
              VideoFrameView(image: videoFrames[currentFrameIndex])
                .padding()

              // Frame counter
              Text(
                "Frame \(currentFrameIndex + 1) / \(videoFrames.count)"
              )
              .padding(.bottom, 5)

              // Playback controls
              HStack {
                Button(action: {
                  currentFrameIndex = max(
                    0,
                    currentFrameIndex - 1
                  )
                }) {
                  Image(systemName: "backward.fill")
                    .font(.title)
                    .padding()
                }
                .disabled(currentFrameIndex == 0)

                Button(action: {
                  togglePlayback()
                }) {
                  Image(
                    systemName: isPlaying
                      ? "pause.fill" : "play.fill"
                  )
                  .font(.title)
                  .padding()
                }

                Button(action: {
                  currentFrameIndex = min(
                    videoFrames.count - 1,
                    currentFrameIndex + 1
                  )
                }) {
                  Image(systemName: "forward.fill")
                    .font(.title)
                    .padding()
                }
                .disabled(
                  currentFrameIndex == videoFrames.count - 1
                )
              }

              // FPS control
              HStack {
                Text("Speed: \(Int(fps)) fps")
                Slider(value: $fps, in: 1...30, step: 1.0)
                  .frame(width: 150)
              }
              .padding(.top, 10)
              .onChange(of: fps) { newValue in
                if isPlaying {
                  // Restart playback with new FPS
                  togglePlayback()
                  togglePlayback()
                }
              }

              // Export Controls
              VStack(spacing: 10) {
                if isExporting {
                  VStack {
                    ProgressView(value: exportProgress, total: 1.0)
                      .progressViewStyle(LinearProgressViewStyle())
                    Text("Exporting: \(Int(exportProgress * 100))%")
                      .font(.caption)
                      .foregroundColor(.blue)
                    if let message = exportMessage {
                      Text(message)
                        .font(.caption)
                        .foregroundColor(.gray)
                    }
                  }
                  .padding()
                } else {
                  Button(action: {
                    saveVideoToPhotoLibrary()
                  }) {
                    HStack {
                      Image(systemName: "photo.badge.plus")
                      Text("Save to Photos")
                    }
                  }
                  .padding(.horizontal, 15)
                  .padding(.vertical, 8)
                  .background(Color.green)
                  .foregroundColor(.white)
                  .cornerRadius(8)
                }
              }
              .padding(.top, 15)

            }
          }
          // ---------------- END: Conditional Display ----------------
        }
      }
      .padding()
      .navigationTitle("Mobile I2V")
      .navigationBarItems(
        trailing:
          Button("Reset") {
            // Reset the view
            videoFrames = []
            currentFrameIndex = 0
            isPlaying = false
            timer?.invalidate()
            timer = nil
            selectedImage = nil
            imageSelection = nil
            errorMessage = nil
            flowScore = 0.6  // Reset flow score
            seed = 42  // Reset seed
          }
          .disabled(isLoading || isExporting || (videoFrames.isEmpty && selectedImage == nil))
      )
      .onDisappear {
        // Stop the timer when the view disappears
        timer?.invalidate()
        timer = nil
      }
      .alert("Export Successful", isPresented: $showingExportSuccess) {
        Button("OK") {
          showingExportSuccess = false
          exportMessage = nil
        }
      } message: {
        Text(exportMessage ?? "Video exported successfully")
      }
    }
  }

  private func togglePlayback() {
    isPlaying.toggle()

    if isPlaying {
      // Start playback with the current FPS
      timer = Timer.scheduledTimer(
        withTimeInterval: 1.0 / fps,
        repeats: true
      ) { _ in
        if currentFrameIndex < videoFrames.count - 1 {
          currentFrameIndex += 1
        } else {
          // Loop back to the beginning
          currentFrameIndex = 0
        }
      }
    } else {
      // Stop playback
      timer?.invalidate()
      timer = nil
    }
  }

  @MainActor  // UI 更新都在主线程，避免额外切线程
  private func generateVideo() async {
    isLoading = true
    errorMessage = nil
    videoFrames = []
    currentFrameIndex = 0
    isPlaying = false
    timer?.invalidate()
    timer = nil
    defer { isLoading = false }

    do {
      // 确保有选择的图像
      guard let selectedImage = selectedImage else {
        throw NSError(
          domain: "DataLoading",
          code: 1,
          userInfo: [
            NSLocalizedDescriptionKey:
              "Please select a first frame image"
          ]
        )
      }

      // 将UIImage转换为CGImage
      // Redrawing applies UIImage's EXIF orientation before Core Graphics preprocessing.
      let normalized = UIGraphicsImageRenderer(size: selectedImage.size).image { _ in
        selectedImage.draw(in: CGRect(origin: .zero, size: selectedImage.size))
      }
      guard let firstFrameCGImage = normalized.cgImage else {
        throw NSError(
          domain: "ImageProcessing",
          code: 2,
          userInfo: [
            NSLocalizedDescriptionKey:
              "Unable to process selected image"
          ]
        )
      }

      videoFrames = try await VideoPipeline.shared.generate(
        image: firstFrameCGImage, seed: seed, flowScore: Float(flowScore))
    } catch {
      // Handle errors
      DispatchQueue.main.async {
        errorMessage = error.localizedDescription
        isLoading = false
      }
    }
  }

  private func saveVideoToPhotoLibrary() {
    isExporting = true
    exportProgress = 0.0
    exportMessage = "Saving to Photo Library..."

    VideoExportService.shared.saveVideoToPhotoLibrary(
      frames: videoFrames,
      fps: fps,
      progressHandler: { progress in
        DispatchQueue.main.async {
          self.exportProgress = progress
        }
      },
      completionHandler: { result in
        DispatchQueue.main.async {
          self.isExporting = false

          switch result {
          case .success():
            self.exportMessage = "Video saved to Photo Library successfully!"
            self.showingExportSuccess = true
          case .failure(let error):
            self.errorMessage = error.localizedDescription
          }
        }
      }
    )
  }
}

struct VideoFrameView: View {
  let image: CGImage

  var body: some View {
    Image(decorative: image, scale: 1.0)
      .resizable()
      .aspectRatio(contentMode: .fit)
      .cornerRadius(8)
  }
}

#Preview {
  VideoContentView()
}
