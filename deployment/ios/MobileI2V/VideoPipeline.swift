// Extracted from MobileSana's on-device MobileI2V pipeline (Silver Yu, 2025).
import CoreGraphics
import CoreML
import Foundation

enum PipelineError: LocalizedError {
  case missingModel(String)
  case incompatibleModel(String)
  case invalidImage

  var errorDescription: String? {
    switch self {
    case .missingModel(let name):
      return
        "Missing \(name).mlmodelc. Add Transformer, VAEEncoder and VAEDecoder packages to the app's Models folder and rebuild."
    case .incompatibleModel(let details):
      return "Incompatible Core ML model: \(details)"
    case .invalidImage:
      return "Unable to convert the image."
    }
  }
}

/// The distilled checkpoint uses exactly two steps; this is not a generic scheduler.
enum FlowEuler {
  static let timesteps: [Float] = [1000, 760]
  static let sigmas: [Float] = [1, 0.76, 0]

  static func step(
    prediction: MLShapedArray<Float>, sample: MLShapedArray<Float>, index: Int
  ) throws -> MLShapedArray<Float> {
    guard prediction.shape == sample.shape, timesteps.indices.contains(index) else {
      throw PipelineError.incompatibleModel("denoiser output shape or step index")
    }
    let delta = sigmas[index + 1] - sigmas[index]
    return MLShapedArray(
      scalars: zip(sample.scalars, prediction.scalars).map { $0 + $1 * delta },
      shape: sample.shape)
  }

  static func condition(
    _ sample: MLShapedArray<Float>, firstFrame: MLShapedArray<Float>
  ) throws -> MLShapedArray<Float> {
    let shape = sample.shape
    guard shape.count == 5, shape[0] == 1,
      firstFrame.shape == [1, shape[1], 1, shape[3], shape[4]]
    else { throw PipelineError.incompatibleModel("first-frame latent shape") }
    let plane = shape[3] * shape[4]
    var values = sample.scalars
    let first = firstFrame.scalars
    for channel in 0..<shape[1] {
      let start = channel * shape[2] * plane
      values.replaceSubrange(
        start..<start + plane, with: first[channel * plane..<(channel + 1) * plane])
    }
    return MLShapedArray(scalars: values, shape: shape)
  }
}

/// Model loading and synchronous image conversion run off the main actor.
actor VideoPipeline {
  static let shared = VideoPipeline()
  static let modelNames = ["Transformer", "VAEEncoder", "VAEDecoder"]
  static let latentShape = [1, 128, 3, 23, 40]
  private var models: [MLModel] = []

  static func modelURL(_ name: String, bundle: Bundle = .main) throws -> URL {
    guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
      throw PipelineError.missingModel(name)
    }
    return url
  }

  private func loadModels() async throws {
    guard models.isEmpty else { return }
    // Resolve all resources before loading any large model.
    let urls = try Self.modelNames.map { try Self.modelURL($0) }
    let configuration = MLModelConfiguration()
    #if targetEnvironment(simulator)
      configuration.computeUnits = .cpuOnly
    #else
      configuration.computeUnits = .cpuAndNeuralEngine
    #endif
    var loaded: [MLModel] = []
    for url in urls {
      loaded.append(try await MLModel.load(contentsOf: url, configuration: configuration))
    }
    try validate(
      loaded[0],
      inputs: [
        "hidden_states": Self.latentShape, "timestep": [1],
        "cond_mask": [1, 3 * 23 * 40], "flow_score": [1],
      ], output: "sample", shape: Self.latentShape)
    try validate(
      loaded[1], inputs: ["latents": [1, 3, 1, 736, 1280]],
      output: "sample0", shape: [1, 128, 1, 23, 40])
    try validate(
      loaded[2], inputs: ["latents": Self.latentShape],
      output: "sample", shape: [1, 3, 17, 736, 1280])
    models = loaded
  }

  private func validate(
    _ model: MLModel, inputs: [String: [Int]], output: String, shape: [Int]
  ) throws {
    let description = model.modelDescription
    guard Set(description.inputDescriptionsByName.keys) == Set(inputs.keys) else {
      throw PipelineError.incompatibleModel(
        "input names; legacy transformer packages are not supported")
    }
    for (name, expected) in inputs.merging([output: shape], uniquingKeysWith: { $1 }) {
      let feature =
        inputs[name] == nil
        ? description.outputDescriptionsByName[name] : description.inputDescriptionsByName[name]
      guard let constraint = feature?.multiArrayConstraint,
        constraint.shape.map({ $0.intValue }) == expected, constraint.dataType == .float32
      else { throw PipelineError.incompatibleModel("\(name) must be float32 \(expected)") }
    }
  }

  private func predict(
    _ model: MLModel, inputs: [String: MLShapedArray<Float>], output: String
  ) async throws -> MLShapedArray<Float> {
    let provider = try MLDictionaryFeatureProvider(
      dictionary: inputs.mapValues { MLMultiArray($0) })
    let result = try await model.prediction(from: provider)
    guard let array = result.featureValue(for: output)?.multiArrayValue else {
      throw PipelineError.incompatibleModel("missing output \(output)")
    }
    return MLShapedArray<Float>(array)
  }

  func generate(image: CGImage, seed: UInt64, flowScore: Float) async throws -> [CGImage] {
    try await loadModels()
    let first = try await predict(
      models[1], inputs: ["latents": Self.prepareImage(image)], output: "sample0")
    let noise = await MLTensor(
      randomNormal: Self.latentShape, mean: Float(0), standardDeviation: Float(1),
      seed: seed, scalarType: Float.self
    ).shapedArray(of: Float.self)
    var latents = try FlowEuler.condition(noise, firstFrame: first)
    let plane = 23 * 40
    let mask = MLShapedArray<Float>(
      scalars: Array(repeating: 1, count: plane) + Array(repeating: 0, count: 2 * plane),
      shape: [1, 3 * plane])
    for (index, timestep) in FlowEuler.timesteps.enumerated() {
      let prediction = try await predict(
        models[0],
        inputs: [
          "hidden_states": latents, "timestep": MLShapedArray(scalars: [timestep], shape: [1]),
          "cond_mask": mask, "flow_score": MLShapedArray(scalars: [flowScore], shape: [1]),
        ], output: "sample")
      latents = try FlowEuler.condition(
        FlowEuler.step(prediction: prediction, sample: latents, index: index), firstFrame: first)
    }
    // Preserve the original decoder-space latent convention: no extra VAE scaling.
    let video = try await predict(models[2], inputs: ["latents": latents], output: "sample")
    return try Self.frames(from: video, cropHeight: 720)
  }

  static func prepareImage(_ image: CGImage, width: Int = 1280, height: Int = 736) throws
    -> MLShapedArray<Float>
  {
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
      guard
        let context = CGContext(
          data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
          bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        )
      else { return false }
      context.interpolationQuality = .high
      context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
      return true
    }
    guard drawn else { throw PipelineError.invalidImage }
    let plane = width * height
    var values = [Float](repeating: 0, count: plane * 3)
    for channel in 0..<3 {
      for pixel in 0..<plane {
        values[channel * plane + pixel] = Float(bytes[pixel * 4 + channel]) / 127.5 - 1
      }
    }
    return MLShapedArray(scalars: values, shape: [1, 3, 1, height, width])
  }

  /// Decode NCTHW, not the frame-first layout of the old experimental packages.
  static func frames(from video: MLShapedArray<Float>, cropHeight: Int) throws -> [CGImage] {
    let shape = video.shape
    guard shape.count == 5, shape[0] == 1, shape[1] == 3, shape.allSatisfy({ $0 > 0 }),
      cropHeight > 0
    else { throw PipelineError.incompatibleModel("decoder output must be [1, 3, T, H, W]") }
    let values = video.scalars
    let time = shape[2]
    let height = shape[3]
    let width = shape[4]
    let visibleHeight = min(height, cropHeight)
    var result: [CGImage] = []
    for frame in 0..<time {
      var bytes = [UInt8](repeating: 255, count: visibleHeight * width * 4)
      for pixel in 0..<visibleHeight * width {
        for channel in 0..<3 {
          let value = values[(channel * time + frame) * height * width + pixel]
          guard value.isFinite else {
            throw PipelineError.incompatibleModel("non-finite decoder output")
          }
          bytes[pixel * 4 + channel] = UInt8(min(255, max(0, (value + 1) * 127.5)).rounded())
        }
      }
      guard let provider = CGDataProvider(data: Data(bytes) as CFData),
        let image = CGImage(
          width: width, height: visibleHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
      else { throw PipelineError.invalidImage }
      result.append(image)
    }
    return result
  }
}
