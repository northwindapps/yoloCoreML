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
        previewLayer.isHidden = true
        //view.layer.addSublayer(previewLayer)

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

        // Fix the orientation of the image
        let fixedImage = image
        let imageSize = fixedImage.size

        UIGraphicsBeginImageContextWithOptions(imageSize, false, 0.0)
        fixedImage.draw(in: CGRect(origin: .zero, size: imageSize))

        for observation in results {
            let boundingBox = observation.boundingBox
            let rect = CGRect(
                x: boundingBox.origin.x * imageSize.width,
                y: (1 - boundingBox.origin.y - boundingBox.height) * imageSize.height,
                width: boundingBox.width * imageSize.width,
                height: boundingBox.height * imageSize.height
            )

            UIColor.red.setStroke()
            UIRectFrame(rect)

            if let label = observation.labels.first?.identifier {
                let textRect = CGRect(x: rect.origin.x, y: rect.origin.y - 20, width: rect.size.width, height: 20)
                let text = NSAttributedString(string: label, attributes: [.foregroundColor: UIColor.red])
                text.draw(in: textRect)
            }
        }

        let annotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        DispatchQueue.main.async {
            self.imageView.image = annotatedImage
        }
    }


    func convertCIImageToUIImage(ciImage: CIImage) -> UIImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            return UIImage(cgImage: cgImage)
        }
        return nil
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        connection.videoOrientation = AVCaptureVideoOrientation.portrait
        let currentTime = Date()
        if currentTime.timeIntervalSince(lastPredictionTime) < 1.0 {
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

