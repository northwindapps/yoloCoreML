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
    var capturedImages:[UIImage] = []
    var bottomLabel = UILabel()
    var actionButton = UIButton(type: .system)
    var dateSlashes = [String]()
    var totalValues = [String]()
    var tvCoordinates = [CGRect]()
    var totalLabelCoordinates = [CGRect]()
    var shopNames = [String]()
    var counter = 5
    var maxLimit = 5

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
        
        
        // Create and configure the label
        bottomLabel.text = "Ready to go.."
        bottomLabel.textAlignment = .left
        bottomLabel.textColor = .white
        bottomLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        bottomLabel.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        bottomLabel.numberOfLines = 1
        bottomLabel.layer.zPosition = 2
        
        // Set the frame for the label at the bottom
        bottomLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(bottomLabel)
        
        // Add constraints for the label
        NSLayoutConstraint.activate([
            bottomLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            bottomLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            bottomLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            bottomLabel.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Create and configure the button
        actionButton.setTitle("Scan", for: .normal)
        actionButton.setTitleColor(.white, for: .normal)
        actionButton.backgroundColor = UIColor.systemBlue
        actionButton.layer.cornerRadius = 8
        actionButton.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
    
        actionButton.layer.zPosition = 2
        
        // Set the frame for the button
        actionButton.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(actionButton)
        
        // Add constraints for the button
        NSLayoutConstraint.activate([
            actionButton.centerXAnchor.constraint(equalTo: bottomLabel.rightAnchor, constant: -50),
            actionButton.centerYAnchor.constraint(equalTo: bottomLabel.centerYAnchor),
            actionButton.widthAnchor.constraint(equalToConstant: 100),
            actionButton.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    @objc func buttonTapped() {
        print("Button tapped!")
        actionButton.setTitle("Scanning...", for: .normal)
        totalValues.removeAll()
        dateSlashes.removeAll()
        
        // Add your button action here
        counter = 0
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
        previewLayer.isHidden = false
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
        
        if counter > maxLimit{
            print("Reached the limit")
//            performOCR(on: image) { recognizedText in
//                if let text = recognizedText {
//                    print("Image: \(text)")
//                } else {
//                    print("No text recognized")
//                }
//            }
            return
        }

        // Fix the orientation of the image
        let fixedImage = image
        let imageSize = fixedImage.size

        UIGraphicsBeginImageContextWithOptions(imageSize, false, 0.0)
        fixedImage.draw(in: CGRect(origin: .zero, size: imageSize))

        //save image
        //capturedImages.append(fixedImage)
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
            
            //
            // Convert UIImage to CGImage
            guard let cgImage = fixedImage.cgImage else { return }
            
            // Crop the image using the rect
            guard let croppedCGImage = cgImage.cropping(to: rect) else { return }
            
            // Convert cropped CGImage back to UIImage
            let croppedImage = UIImage(cgImage: croppedCGImage)
            //capturedImages.append(croppedImage)
            
            if let label = observation.labels.first?.identifier {
                let image = croppedImage
                performOCR(on: image) { recognizedText in
                    if let text = recognizedText {
                        print("Key:\(label),Value: \(text)")
                        if label == "tv"{
                            let filteredString = self.filterDigits(inputString: text.first ?? "")
                            if Double(filteredString) != nil{
                                self.totalValues.append(text.first!)
                            }
                        }
                        if label == "datesla"{
                            if self.filterDateWithSlashFormat(inputString: text.first ?? ""){
                                self.dateSlashes.append(text.first!)
                            }
                        }
                        self.counter += 1
                    } else {
                        print("No text recognized")
                    }
                }
            }
        }
    }
    
    func filterDigits(inputString:String)->String{
        let pattern = "[0-9.]+"
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let matches = regex.matches(in: inputString, range: NSRange(inputString.startIndex..., in: inputString))
            let numbers = matches.map { match -> String in
                let range = Range(match.range, in: inputString)!
                return String(inputString[range])
            }
            print(numbers) // Output: ["123.45", "01.01.2025"]
            return numbers.first ?? ""
        }
        return ""
    }
    
    func filterDateWithSlashFormat(inputString:String)->Bool{
        if inputString.contains("/"){
            let numArys = inputString.components(separatedBy: "/")
            if numArys.count == 3{
                if Double(numArys[0]) != nil && Double(numArys[1]) != nil && Double(numArys[2]) != nil{
                    return true
                }
            }
        }
        if inputString.contains("-"){
            let numArys = inputString.components(separatedBy: "-")
            if numArys.count == 3{
                if Double(numArys[0]) != nil && Double(numArys[1]) != nil && Double(numArys[2]) != nil{
                    return true
                }
            }
        }
        return false
    }

    func performOCR(on image: UIImage, completion: @escaping ([String]?) -> Void) {
        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else {
            completion(nil)
            return
        }

        // Create a request for text recognition
        let request = VNRecognizeTextRequest { (request, error) in
            if let error = error {
                print("Error during OCR: \(error)")
                completion(nil)
                return
            }

            // Extract recognized text
            let recognizedStrings = request.results?.compactMap { result -> String? in
                guard let observation = result as? VNRecognizedTextObservation else { return nil }
                return observation.topCandidates(1).first?.string
            }

            completion(recognizedStrings)
        }

        // Set recognition level and language (optional)
        request.recognitionLevel = .accurate // .accurate or .fast
        request.recognitionLanguages = ["en-US"] // You can add more languages

        // Create a request handler
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        // Perform the request
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try requestHandler.perform([request])
            } catch {
                print("Error performing OCR request: \(error)")
                completion(nil)
            }
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
        
//        Run prediction in background thread to avoid lag
        DispatchQueue.global(qos: .userInitiated).async {
            self.predict(image: uiImage)
            let dateText = self.dateSlashes.first ?? "no date"
            var totalText = self.totalValues.first ?? "no total"
            for (i,each) in self.totalValues.enumerated(){
                if each.contains("$"){
                    totalText = each
                }
            }
            // Update UI on the main thread
            DispatchQueue.main.async {
                self.bottomLabel.text = "\(dateText) \(",") \(totalText)"
                if self.maxLimit > self.counter{
                    self.actionButton.setTitle("Retry", for: .normal)
                }
            }
        }
    }
}

