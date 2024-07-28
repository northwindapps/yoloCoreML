import CoreML
import Vision
import UIKit
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var model: VNCoreMLModel?
    var imageView: UIImageView!
    var shapeLayers: [CAShapeLayer] = []
    var textLayers: [CATextLayer] = []
    var lastPredictionTime: Date = Date()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadModel()
        setupCamera()
    }

    func setupUI() {
        imageView = UIImageView(frame: view.bounds)
        imageView.contentMode = .scaleAspectFit
        imageView.layer.zPosition = 1
        view.addSubview(imageView)
    }

    func loadModel() {
        do {
            guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") else {
                fatalError("Failed to find the model file.")
            }
            let coreMLModel = try MLModel(contentsOf: modelURL)
            self.model = try VNCoreMLModel(for: coreMLModel)
        } catch {
            fatalError("Failed to load model: \(error.localizedDescription)")
        }
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo

        guard let camera = AVCaptureDevice.default(for: .video) else { return }
        let cameraInput = try? AVCaptureDeviceInput(device: camera)

        if captureSession.canAddInput(cameraInput!) {
            captureSession.addInput(cameraInput!)
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        captureSession.startRunning()
    }

    func predict(image: UIImage) {
        guard let ciImage = CIImage(image: image), let model = model else {
            fatalError("Unable to create CIImage from UIImage or model not loaded")
        }

        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                print("Failed to perform request: \(error.localizedDescription)")
                return
            }
            self.processResults(request.results, in: image)
        }

        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform request: \(error.localizedDescription)")
        }
    }

    func processResults(_ results: [Any]?, in image: UIImage) {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            print("No results or results are of unexpected type")
            return
        }

        DispatchQueue.main.async {
            self.clearPreviousShapes()

            let imageSize = image.size
            let imageViewSize = self.imageView.bounds.size

            for observation in results {
                let boundingBox = observation.boundingBox
                let rect = CGRect(
                    x: boundingBox.origin.x * imageViewSize.width,
                    y: (1 - boundingBox.origin.y - boundingBox.height) * imageViewSize.height,
                    width: boundingBox.width * imageViewSize.width,
                    height: boundingBox.height * imageViewSize.height
                )

                let shapeLayer = CAShapeLayer()
                shapeLayer.path = UIBezierPath(rect: rect).cgPath
                shapeLayer.strokeColor = UIColor.red.cgColor
                shapeLayer.lineWidth = 2.0
                shapeLayer.fillColor = UIColor.clear.cgColor

                self.imageView.layer.addSublayer(shapeLayer)
                self.shapeLayers.append(shapeLayer)

                if let label = observation.labels.first?.identifier {
                    let textLayer = CATextLayer()
                    textLayer.string = label
                    textLayer.foregroundColor = UIColor.red.cgColor
                    textLayer.fontSize = 14
                    textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y - 20, width: rect.size.width, height: 20)
                    textLayer.contentsScale = UIScreen.main.scale
                    self.imageView.layer.addSublayer(textLayer)
                    self.textLayers.append(textLayer)
                }
            }
        }
    }

    func clearPreviousShapes() {
        for layer in shapeLayers {
            layer.removeFromSuperlayer()
        }
        shapeLayers.removeAll()
        
        for layer in textLayers {
            layer.removeFromSuperlayer()
        }
        textLayers.removeAll()
    }

    func convertCIImageToUIImage(ciImage: CIImage) -> UIImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            return UIImage(cgImage: cgImage)
        }
        return nil
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        let currentTime = Date()
        if currentTime.timeIntervalSince(lastPredictionTime) < 3.0 {
            return
        }
        lastPredictionTime = currentTime
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let uiImage = convertCIImageToUIImage(ciImage: ciImage) else {
            return
        }
        
        DispatchQueue.main.async {
            self.predict(image: uiImage)
        }
        
    }
}

