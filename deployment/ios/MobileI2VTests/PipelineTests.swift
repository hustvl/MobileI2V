import CoreGraphics
import CoreML
import XCTest

@testable import MobileI2V

final class PipelineTests: XCTestCase {
  func testTwoStepConditionedDenoising() throws {
    XCTAssertEqual(FlowEuler.timesteps, [1000, 760])
    let shape = [1, 2, 3, 1, 2]
    let first = MLShapedArray<Float>(scalars: [10, 11, 20, 21], shape: [1, 2, 1, 1, 2])
    let sample = MLShapedArray<Float>(scalars: Array(0..<12).map(Float.init), shape: shape)
    var latents = try FlowEuler.condition(sample, firstFrame: first)
    XCTAssertEqual(latents.scalars, [10, 11, 2, 3, 4, 5, 20, 21, 8, 9, 10, 11])
    let prediction = MLShapedArray<Float>(scalars: Array(repeating: 2, count: 12), shape: shape)
    latents = try FlowEuler.condition(
      FlowEuler.step(prediction: prediction, sample: latents, index: 0), firstFrame: first)
    XCTAssertEqual(latents.scalars[2], 1.52, accuracy: 1e-5)
    latents = try FlowEuler.condition(
      FlowEuler.step(prediction: prediction, sample: latents, index: 1), firstFrame: first)
    let expected: [Float] = [10, 11, 0, 1, 2, 3, 20, 21, 6, 7, 8, 9]
    for (actual, reference) in zip(latents.scalars, expected) {
      XCTAssertEqual(actual, reference, accuracy: 1e-5)
    }
    XCTAssertThrowsError(try FlowEuler.condition(sample, firstFrame: sample))
    XCTAssertThrowsError(try FlowEuler.step(prediction: first, sample: sample, index: 0))
  }

  func testNCTHWConversionCropAndImageNormalization() throws {
    // Two different frames, three different channels, non-square image and a discarded row.
    let values: [Float] = [
      -1, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1,
      0, 1, -1, 1, 1, 1, -1, 1, 0, -1, -1, -1,
      1, -1, 0, 1, 1, 1, 0, -1, 1, -1, -1, -1,
    ]
    let video = MLShapedArray(scalars: values, shape: [1, 3, 2, 2, 3])
    let frames = try VideoPipeline.frames(from: video, cropHeight: 1)
    XCTAssertEqual(frames.count, 2)
    XCTAssertEqual(frames[0].width, 3)
    XCTAssertEqual(frames[0].height, 1)
    let first = Array(try XCTUnwrap(frames[0].dataProvider?.data) as Data)
    let second = Array(try XCTUnwrap(frames[1].dataProvider?.data) as Data)
    XCTAssertEqual(first, [0, 128, 255, 255, 128, 255, 0, 255, 255, 0, 128, 255])
    XCTAssertEqual(second, [255, 0, 128, 255, 128, 255, 0, 255, 0, 128, 255, 255])
    let encoded = try VideoPipeline.prepareImage(frames[0], width: 3, height: 1)
    XCTAssertEqual(encoded.shape, [1, 3, 1, 1, 3])
    let expected: [Float] = [-1, 1 / 255, 1, 1 / 255, 1, -1, 1, -1, 1 / 255]
    for (actual, reference) in zip(encoded.scalars, expected) {
      XCTAssertEqual(actual, reference, accuracy: 1e-5)
    }
    XCTAssertThrowsError(
      try VideoPipeline.frames(
        from: MLShapedArray(scalars: values, shape: [2, 1, 3, 2, 3]), cropHeight: 1))
  }

  func testMissingModelIsActionable() {
    XCTAssertThrowsError(try VideoPipeline.modelURL("AbsentModel", bundle: Bundle(for: Self.self)))
    {
      XCTAssertTrue($0.localizedDescription.contains("AbsentModel.mlmodelc"))
      XCTAssertTrue($0.localizedDescription.contains("Models folder"))
    }
  }
}
