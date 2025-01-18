import CoreML
import Vision
import UIKit
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var model: VNCoreMLModel?
    var model2: VNCoreMLModel?
    var model3: VNCoreMLModel?
    var imageView: UIImageView!
    var shapeLayers: [CAShapeLayer] = []
    var textLayers: [CATextLayer] = []
    var lastPredictionTime: Date = Date()
    var capturedImages:[UIImage] = []
    var bottomLabel = UILabel()
    var actionButton = UIButton(type: .system)
    var dateSlashes = [String]()
    var totalValues = [String]()
    var shopNames = [String]()
    var totalPool = [String]()
    var datePool = [String]()
    var shopPool = [String]()
    var counter = 5
    var counter2 = 5
    var counter3 = 5
    var maxLimit = 5
    var Menuview:Menu!
    var textFields = [UITextField]()
    var repoDictionary = [[String: String]]()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadModel()
        setupCamera()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        askToLoadData()
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
        bottomLabel.isUserInteractionEnabled = true
        
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
        // Step 3: Add the tap gesture recognizer
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(labelTapped))
        bottomLabel.addGestureRecognizer(tapGesture)
        
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
    
    func askToLoadData() {
        // Check if there is saved data
        if let savedArray = UserDefaults.standard.array(forKey: "repoDictionary") as? [[String: String]] {
            // Create an alert controller
            let alert = UIAlertController(
                title: "Load Data",
                message: "Do you want to load the saved data?",
                preferredStyle: .alert
            )
            
            // Add "Load" action
            alert.addAction(UIAlertAction(title: "Load", style: .default, handler: { _ in
                self.repoDictionary = savedArray
                print("Data loaded: \(self.repoDictionary)")
            }))
            
            // Add "Cancel" action
            alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { _ in
                print("Load canceled")
            }))
            
            // Present the alert
            self.present(alert, animated: true)
            
        } else {
            print("No saved data to load.")
        }
    }

    
    @objc func buttonTapped() {
        print("Button tapped!")
        actionButton.setTitle("Scanning...", for: .normal)
        totalValues.removeAll()
        dateSlashes.removeAll()
        shopNames.removeAll()
        actionButton.isEnabled = false
        
        // Add your button action here
        counter = 0
        counter2 = 0
        counter3 = 0
    }
    
    @objc func labelTapped() {
        let mergedPool = self.datePool + self.shopPool + self.totalPool
        print("Label tapped!")
        //showMenu
        if Menuview != nil{
            if self.view.subviews.contains(Menuview){
                Menuview.removeFromSuperview()
            }else{
                Menuview = Menu(frame: CGRect(x:Int(5),y:Int(self.view.frame.height - 600), width: 300,height: 500))
                self.view.addSubview(Menuview)
            }
        }else{
            Menuview = Menu(frame: CGRect(x:Int(5),y:Int(self.view.frame.height - 600), width: 300,height: 500))
//            Menuview.hintCloseButton.addTarget(self, action: #selector(ViewController.closeHview), for: UIControl.Event.touchUpInside)
            
            self.view.addSubview(Menuview)
        }
        //
        Menuview.backgroundColor = UIColor.white.withAlphaComponent(0.5)
        Menuview.backgroundColor = UIColor.white.withAlphaComponent(0.5)

        let shopText = self.shopNames.first ?? "no shop"
        let dateText = self.dateSlashes.first ?? "no date"
        var totalText = self.totalValues.first ?? "no total"
        textFields = []
        // Create the input fields
        for i in 0..<3 {
            let textField = UITextField(frame: CGRect(x: 10, y: 20 + i * 60, width: 280, height: 20))
            textFields.append(textField)
            if i == 0{
                textField.placeholder = "date" // Placeholder text
                textField.text = dateText
            }
            if i == 1{
                textField.placeholder = "shop" // Placeholder text
                textField.text = shopText
            }
            if i == 2{
                textField.placeholder = "total" // Placeholder text
                textField.text = totalText
            }
            
            textField.borderStyle = .roundedRect // Add rounded corners
            textField.backgroundColor = .white // Set background color
            textField.textColor = .black // Set text color
            textField.font = UIFont.systemFont(ofSize: 16) // Set font size
            textField.clearButtonMode = .whileEditing // Add clear button
            Menuview.addSubview(textField) // Add the text field to the menu
        }
        
        // Create the UICollectionView layout
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .vertical
        layout.itemSize = CGSize(width: 80, height: 20)
        layout.minimumLineSpacing = 10
        layout.minimumInteritemSpacing = 10
        // Create the UICollectionView
        let collectionView = UICollectionView(frame: CGRect(x: 10, y: 200, width: 280, height: 150), collectionViewLayout: layout)
        collectionView.register(CustomCollectionViewCell.self, forCellWithReuseIdentifier: "CustomCell")
        collectionView.backgroundColor = UIColor.clear
        collectionView.dataSource = self
        collectionView.delegate = self
        collectionView.reloadData()

        Menuview.addSubview(collectionView)

//        // Create the button
//        let button = UIButton(frame: CGRect(x: 10, y: 420, width: 280, height: 50))
//        button.setTitle("Submit", for: .normal)
//        button.backgroundColor = UIColor.systemBlue
//        button.setTitleColor(.white, for: .normal)
//        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
//        button.layer.cornerRadius = 8 // Rounded corners
//        button.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
//
//        Menuview.addSubview(button) // Add the button to the menu
//        self.view.addSubview(Menuview)
        // Button titles and actions
        let buttonTitles = ["Save", "Cancel","Table"]
        let buttonColors: [UIColor] = [.systemBlue, .white, .white]
        let buttonTitleColors: [UIColor] = [.white, .black, .black]
        let buttonActions: [Selector] = [#selector(saveTapped), #selector(cancelTapped), #selector(tableTapped)]

        // Loop to create three buttons
        for (index, title) in buttonTitles.enumerated() {
            let button = UIButton(frame: CGRect(x: 10 + index * 70, // Adjust x position for each button
                                                y: 420,
                                                width: 60, // Width of each button
                                                height: 20))
            button.setTitle(title, for: .normal)
            button.backgroundColor = buttonColors[index] // Set color based on index
            button.setTitleColor(buttonTitleColors[index], for: .normal)
            button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 10)
            button.layer.cornerRadius = 8
            button.addTarget(self, action: buttonActions[index], for: .touchUpInside)
            Menuview.addSubview(button) // Add each button to the menu
        }

        self.view.addSubview(Menuview)
    }
    
    @objc func saveTapped() {
        print("Submit button tapped")
        totalValues.removeAll()
        dateSlashes.removeAll()
        shopNames.removeAll()
        totalPool.removeAll()
        datePool.removeAll()
        shopPool.removeAll()
        let date = textFields[0].text
        let shop = textFields[1].text
        let total = textFields[2].text
        let newEntry: [String: String] = [
            "date": date!,
            "shop": shop!,
            "total": total!
        ]
        repoDictionary.append(newEntry)
        UserDefaults.standard.set(repoDictionary, forKey: "repoDictionary")
        textFields = []
        if Menuview != nil{
            Menuview.removeFromSuperview()
        }
    }

    @objc func cancelTapped() {
        print("Cancel button tapped")
        if Menuview != nil{
            Menuview.removeFromSuperview()
        }
    }

    @objc func resetTapped() {
        print("Reset button tapped")
    }
    
    @objc func tableTapped() {
        print("Table button tapped")
        let targetViewController = self.storyboard!.instantiateViewController( withIdentifier: "tableview" ) as! TableViewController//Landscape
        targetViewController.repoDictionary = repoDictionary
        targetViewController.modalPresentationStyle = .fullScreen
        self.present( targetViewController, animated: true, completion: nil)
    }

    func loadModel() {
        do {
            guard let modelURL = Bundle.main.url(forResource: "shop_name", withExtension: "mlmodelc") else {
                fatalError("Failed to find the model file.")
            }
            let coreMLModel = try MLModel(contentsOf: modelURL)
            self.model = try VNCoreMLModel(for: coreMLModel)
            //
            guard let modelURL2 = Bundle.main.url(forResource: "total", withExtension: "mlmodelc") else {
                fatalError("Failed to find the model file.")
            }
            let coreMLModel2 = try MLModel(contentsOf: modelURL2)
            self.model2 = try VNCoreMLModel(for: coreMLModel2)
            guard let modelURL3 = Bundle.main.url(forResource: "date", withExtension: "mlmodelc") else {
                fatalError("Failed to find the model file.")
            }
            let coreMLModel3 = try MLModel(contentsOf: modelURL3)
            self.model3 = try VNCoreMLModel(for: coreMLModel3)
            
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
        if counter > maxLimit{
            print("Reached the limit1")
            return
        }
        if counter2 > maxLimit{
            print("Reached the limit2")
            return
        }
        if counter3 > maxLimit{
            print("Reached the limit3")
            return
        }
        guard let ciImage = CIImage(image: image), let model = model, let model2 = model2, let model3 = model3 else {
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
        //
        let request2 = VNCoreMLRequest(model: model2) { request2, error in
            if let error = error {
                print("Failed to perform request: \(error.localizedDescription)")
                return
            }
            self.processResults2(request2.results, in: image)
        }
        do {
            try handler.perform([request2])
        } catch {
            print("Failed to perform request: \(error.localizedDescription)")
        }
        //
        let request3 = VNCoreMLRequest(model: model3) { request3, error in
            if let error = error {
                print("Failed to perform request: \(error.localizedDescription)")
                return
            }
            self.processResults3(request3.results, in: image)
        }
        do {
            try handler.perform([request3])
        } catch {
            print("Failed to perform request: \(error.localizedDescription)")
        }
    }


    func processResults(_ results: [Any]?, in image: UIImage) {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            print("No results or results are of unexpected type")
            return
        }
        
        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else { return }

        //save image
        for observation in results {
            autoreleasepool {
                let boundingBox = observation.boundingBox
                let rect = CGRect(
                    x: boundingBox.origin.x * image.size.width,
                    y: (1 - boundingBox.origin.y - boundingBox.height) * image.size.height,
                    width: boundingBox.width * image.size.width,
                    height: boundingBox.height * image.size.height
                )
                // Crop the image using the rect
                guard let croppedCGImage = cgImage.cropping(to: rect) else { return }
                
                // Convert cropped CGImage back to UIImage
                let croppedImage = UIImage(cgImage: croppedCGImage)
                if let label = observation.labels.first?.identifier {
                    let image = croppedImage
                    performOCR(on: image) { recognizedText in
                        if let text = recognizedText {
                            print("Key1:\(label),Value: \(text)")
                            if label == "shop"{
                                let filteredString = text.first ?? ""
                                if filteredString != ""{
                                    if (self.shopNames.firstIndex(of: text.first!) == nil){
                                        self.shopNames.append(text.first!)
                                        self.shopPool.append(text.first!)
                                    }
                                }
                            }
//                            self.counter += 1
                        } else {
                            print("No text recognized")
                        }
                    }
                }
            }
        }
        self.counter = maxLimit + 1
    }
    func processResults2(_ results: [Any]?, in image: UIImage) {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            print("No results or results are of unexpected type")
            return
        }

        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else { return }
        //save image
        for observation in results {
            autoreleasepool {
                let boundingBox = observation.boundingBox
                let rect = CGRect(
                    x: boundingBox.origin.x * image.size.width,
                    y: (1 - boundingBox.origin.y - boundingBox.height) * image.size.height,
                    width: boundingBox.width * image.size.width,
                    height: boundingBox.height * image.size.height
                )
                // Crop the image using the rect
                guard let croppedCGImage = cgImage.cropping(to: rect) else { return }
                
                // Convert cropped CGImage back to UIImage
                let croppedImage = UIImage(cgImage: croppedCGImage)
                if let label = observation.labels.first?.identifier {
                    let image = croppedImage
                    performOCR(on: image) { recognizedText in
                        if let text = recognizedText {
                            print("Key2:\(label),Value: \(text)")
                            if label == "totalPair"{
                                let filteredString = self.filterDigits(inputString: text.last ?? "")
                                if Double(filteredString) != nil{
                                    if (self.totalValues.firstIndex(of: self.numberOnlyString(text: text.last!)) == nil){
                                        self.totalValues.append(self.numberOnlyString(text: text.last!))
                                        self.totalPool.append(self.numberOnlyString(text: text.last!))
                                    }
                                }
                            }
//                            self.counter2 += 1
                        } else {
                            print("No text recognized")
                        }
                    }
                }
            }
        }
        self.counter2 = maxLimit + 1
    }
    func processResults3(_ results: [Any]?, in image: UIImage) {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            print("No results or results are of unexpected type")
            return
        }

        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else { return }
        //save image
        for observation in results {
            autoreleasepool {
                let boundingBox = observation.boundingBox
                let rect = CGRect(
                    x: boundingBox.origin.x * image.size.width,
                    y: (1 - boundingBox.origin.y - boundingBox.height) * image.size.height,
                    width: boundingBox.width * image.size.width,
                    height: boundingBox.height * image.size.height
                )
                // Crop the image using the rect
                guard let croppedCGImage = cgImage.cropping(to: rect) else { return }
                
                // Convert cropped CGImage back to UIImage
                let croppedImage = UIImage(cgImage: croppedCGImage)
                if let label = observation.labels.first?.identifier {
                    let image = croppedImage
                    performOCR(on: image) { recognizedText in
                        if let text = recognizedText {
                            print("Key3:\(label),Value: \(text)")
                            if label == "datestr"{
                                if self.filterDateWithSlashFormat(inputString: text.first ?? ""){
                                    if (self.dateSlashes.firstIndex(of: text.first!) == nil){
                                        self.dateSlashes.append(text.first!)
                                        self.datePool.append(text.first!)
                                    }
                                }
                            }
//                            self.counter2 += 1
                        } else {
                            print("No text recognized")
                        }
                    }
                }
            }
        }
        self.counter3 = maxLimit + 1
    }
    
    @objc func imageSaved(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            print("Error saving image: \(error.localizedDescription)")
        } else {
            print("Image saved successfully to photo album!")
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
    
    func numberOnlyString(text: String) -> String {
        
       let numChars = Set("1234567890.")
        return text.replacingOccurrences(of: "-", with: ".").filter {numChars.contains($0) }
    }
    
    func filterDateWithSlashFormat(inputString:String)->Bool{
        if inputString.contains("/"){
            let numArys = inputString.components(separatedBy: "/")
            if numArys.count > 1{
                if Double(numberOnlyString(text: numArys.last!)) != nil{
                    return true
                }
            }
        }
        if inputString.contains("-"){
            let numArys = inputString.components(separatedBy: "-")
            if numArys.count > 1{
                if  Double(numberOnlyString(text: numArys.last!)) != nil{
                    return true
                }
            }
        }
        if inputString.contains("'"){
            let numArys = inputString.components(separatedBy: "'")
            if numArys.count > 1{
                if  Double(numberOnlyString(text: numArys.last!)) != nil{
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
        
        //Run prediction in background thread to avoid lag
        DispatchQueue.global(qos: .userInitiated).async {
            self.predict(image: uiImage)
            let shopText = self.shopNames.first ?? "no shop"
            let dateText = self.dateSlashes.first ?? "no date"
            let totalText = self.totalValues.first ?? "no total"
            // Update UI on the main thread
            DispatchQueue.main.async {
                self.bottomLabel.text = "\(shopText) \(",") \(dateText) \(",") \(totalText)"
//                if self.maxLimit < self.counter{
                    self.actionButton.setTitle("Retry", for: .normal)
                    self.actionButton.isEnabled = true
//                }
            }
        }
    }
}
extension ViewController: UICollectionViewDataSource, UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return (self.datePool.count + self.shopPool.count + self.totalPool.count) // Number of items in the collection view
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {

        guard let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "CustomCell", for: indexPath) as? CustomCollectionViewCell else {
            return UICollectionViewCell()
        }
        
        // Determine which array the current index belongs to
        if indexPath.row < self.datePool.count {
            // Get item from datePool
            cell.label.text = "d: " + self.datePool[indexPath.row]
        } else if indexPath.row < self.datePool.count + self.shopPool.count {
            // Get item from shopPool
            let shopIndex = indexPath.row - self.datePool.count
            cell.label.text = "s: " + self.shopPool[shopIndex]
        } else {
            // Get item from totalPool
            let totalIndex = indexPath.row - self.datePool.count - self.shopPool.count
            cell.label.text = "t: " + self.totalPool[totalIndex]
        }

        cell.label.numberOfLines = 2
        cell.label.tag = indexPath.row
        cell.backgroundColor = UIColor.white // Set the cell background color
        cell.layer.cornerRadius = 8 // Add rounded corners to the cells
        cell.label.font = UIFont.systemFont(ofSize: 10.0)
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(celllabelTapped(_:)))
            cell.label.isUserInteractionEnabled = true // Make sure interaction is enabled
            cell.label.addGestureRecognizer(tapGesture)
        return cell
    }
    @objc func celllabelTapped(_ sender: UITapGestureRecognizer) {
        if let label = sender.view as? UILabel {
            print("Label tapped: \(label.text ?? "No text")")
            // Handle the tap action here
            if label.text!.contains("d:"){
                textFields[0].text = label.text?.replacingOccurrences(of: "d: ", with: "")
            }
            if label.text!.contains("s:"){
                textFields[1].text = label.text?.replacingOccurrences(of: "s: ", with: "")
            }
            if label.text!.contains("t:"){
                textFields[2].text = label.text?.replacingOccurrences(of: "t: ", with: "")
            }
        }
    }
}
class CustomCollectionViewCell: UICollectionViewCell {
    let label: UILabel = {
        let lbl = UILabel()
        lbl.textColor = .black
        lbl.font = UIFont.systemFont(ofSize: 14)
        lbl.textAlignment = .center
        lbl.translatesAutoresizingMaskIntoConstraints = false
        return lbl
    }()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        contentView.addSubview(label)
        
        // Add constraints to center the label within the cell
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            label.widthAnchor.constraint(lessThanOrEqualTo: contentView.widthAnchor, constant: -10),
            label.heightAnchor.constraint(equalToConstant: 20)
        ])
        
        contentView.layer.cornerRadius = 8
        contentView.layer.masksToBounds = true
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}


