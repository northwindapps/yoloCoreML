import UIKit
import MessageUI

class EmailViewController: UIViewController,UITextFieldDelegate,MFMailComposeViewControllerDelegate {
    
    var capturedImage = UIImage()
    let imageView = UIImageView()
    let buttonContainer = UIView()
    let button1 = UIButton(type: .system)
    let button2 = UIButton(type: .system)
    var data: Data?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupImageView()
    }
    
    func setupUI() {
        view.backgroundColor = .white
        setupBottomButtons()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        let appd : AppDelegate = UIApplication.shared.delegate as! AppDelegate
        super.viewDidAppear(animated)
        
        
    }
    
    func setupImageView() {
        // Create and configure the title label
        let titleLabel = UILabel()
        titleLabel.text = "Would you help the dev to improve this model?"
        titleLabel.numberOfLines = 2
        titleLabel.font = UIFont.boldSystemFont(ofSize: 20)
        titleLabel.textAlignment = .center
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(titleLabel)
        
        // Create and configure the description label
        let descriptionLabel = UILabel()
        descriptionLabel.text = """
        By sharing this receipt image with the dev, the model's accuracy might improve. 
        *The receipt image you send will be integrated into the model's puclic training dataset on Roboflow.
        """
        descriptionLabel.font = UIFont.systemFont(ofSize: 16)
        descriptionLabel.textAlignment = .center
        descriptionLabel.numberOfLines = 0 // Allow for multiline text
        descriptionLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(descriptionLabel)
        
        // Configure the UIImageView
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(imageView)
        
        // Set constraints for the title, description, and image view
        NSLayoutConstraint.activate([
            // Title label constraints
            titleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            titleLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            titleLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            // Description label constraints
            descriptionLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 10),
            descriptionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            descriptionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            // UIImageView constraints
            imageView.topAnchor.constraint(equalTo: descriptionLabel.bottomAnchor, constant: 20),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            imageView.bottomAnchor.constraint(equalTo: buttonContainer.topAnchor, constant: -20)
        ])
        
        // Load the captured image
        imageView.image = capturedImage
    }

    
    func setupBottomButtons() {
        // Add button container
        view.addSubview(buttonContainer)
        buttonContainer.translatesAutoresizingMaskIntoConstraints = false
        buttonContainer.backgroundColor = .lightGray
        
        NSLayoutConstraint.activate([
            buttonContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            buttonContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            buttonContainer.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            buttonContainer.heightAnchor.constraint(equalToConstant: 60)
        ])
        
        // Add buttons
        buttonContainer.addSubview(button1)
        buttonContainer.addSubview(button2)
        buttonContainer.backgroundColor = UIColor.white.withAlphaComponent(0.5)
        
        button1.setTitle("Return", for: .normal)
        button2.setTitle("Share", for: .normal)
//        button2.titleLabel?.font = UIFont.systemFont(ofSize: 12)
//        button2.titleLabel?.numberOfLines = 2
        button1.setTitleColor(.gray, for: .normal)
        button2.setTitleColor(.gray, for: .normal)
        styleButton(button: button1)
        styleButton(button: button2)
        
        button1.translatesAutoresizingMaskIntoConstraints = false
        button2.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Button 1 constraints
            button1.centerYAnchor.constraint(equalTo: buttonContainer.centerYAnchor),
            button1.leadingAnchor.constraint(equalTo: buttonContainer.leadingAnchor, constant: 20),
            
            // Button 2 constraints (adjacent to Button 1)
            button2.centerYAnchor.constraint(equalTo: buttonContainer.centerYAnchor),
            button2.leadingAnchor.constraint(equalTo: button1.trailingAnchor, constant: 10),
            button1.widthAnchor.constraint(equalToConstant: 120),
            button2.widthAnchor.constraint(equalToConstant: 120)
        ])
        
        button1.addTarget(self, action: #selector(button1Tapped), for: .touchUpInside)
        button2.addTarget(self, action: #selector(button2Tapped), for: .touchUpInside)
    }
    
    func styleButton(button: UIButton) {
        button.layer.cornerRadius = 15 // Adjust for more or less rounding
        button.layer.borderWidth = 1   // Border thickness
        button.layer.borderColor = UIColor.gray.cgColor // Border color
        button.clipsToBounds = true   // Ensures content stays within rounded edges
    }

    
    @objc func button1Tapped() {
        print("Button 1 tapped")
        if let presentingVC = presentingViewController as? ViewController {
                dismiss(animated: true, completion: nil) // Return to the existing view
        } else {
            let targetViewController = storyboard!.instantiateViewController(withIdentifier: "scanview") as! ViewController
            targetViewController.modalPresentationStyle = .fullScreen
            present(targetViewController, animated: true, completion: nil)
        }
    }
    
    @objc func button2Tapped() {
        print("Button 2 tapped")
        uiImageEmail()
    }
    
    func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
        controller.dismiss(animated: true)
    }
    
    @objc func uiImageEmail() {
        if MFMailComposeViewController.canSendMail() {
            let today: Date = Date()
            let dateFormatter: DateFormatter = DateFormatter()
            dateFormatter.dateFormat = "MM-dd-yyyy HH:mm"
            let date = dateFormatter.string(from: today)
            let mail = MFMailComposeViewController()
            mail.mailComposeDelegate = self
            mail.setSubject("from Total Tracker")
            mail.setToRecipients(["totaltrackerapp@gmail.com"])
            
            // Convert capturedImage to Data
            let imageData = capturedImage.jpegData(compressionQuality: 1.0)
            if imageData != nil{
                mail.addAttachmentData(imageData!, mimeType: "image/jpeg", fileName: "total_tracker_\(date).jpeg")
                present(mail, animated: true, completion: nil)
            }
        }
    }
    
    
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder() // Dismiss the keyboard
        return true
    }
    
}
    
   
