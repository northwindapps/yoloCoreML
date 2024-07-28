import CoreML
import Vision
import UIKit

class ViewController: UIViewController {
    
    var model: VNCoreMLModel?
    var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadModel()
        guard let imageURL = Bundle.main.url(forResource: "billJP", withExtension: "jpg") else {
            fatalError("Failed to find the image file.")
        }
        predictFromURL(imageURL)
    }



    func setupUI() {
            imageView = UIImageView(frame: view.bounds)
            imageView.contentMode = .scaleAspectFit
            view.addSubview(imageView)
        }

    func loadModel() {
        // Load Core ML model
        guard let coreMLModel = try? VNCoreMLModel(for: best().model) else {
            fatalError("Failed to load model")
        }
        self.model = coreMLModel
    }

    func predictFromURL(_ url: URL) {
        guard let image = UIImage(contentsOfFile: url.path) else {
            fatalError("Failed to create UIImage from file.")
    }

        DispatchQueue.global().async {
            DispatchQueue.main.async {
                self.imageView.image = image
                self.predict(image: image)
            }
        }
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

        let imageSize = image.size
        let imageViewSize = imageView.bounds.size
        let scaleX = imageViewSize.width / imageSize.width
        let scaleY = imageViewSize.height / imageSize.height

        UIGraphicsBeginImageContextWithOptions(imageViewSize, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: imageViewSize))

        for observation in results {
            let boundingBox = observation.boundingBox
            let rect = CGRect(
                x: boundingBox.origin.x * imageViewSize.width,
                y: (1 - boundingBox.origin.y - boundingBox.height) * imageViewSize.height,
                width: boundingBox.width * imageViewSize.width,
                height: boundingBox.height * imageViewSize.height
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
}
