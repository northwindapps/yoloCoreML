import UIKit
import MessageUI

class TableViewController: UIViewController,MFMailComposeViewControllerDelegate {
    let tableView = UITableView()
    let buttonContainer = UIView()
    let button1 = UIButton(type: .system)
    let button2 = UIButton(type: .system)
    var repoDictionary = [[String: String]]()
    var data: Data?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    func setupUI() {
        view.backgroundColor = .white
        
        // Setup Bottom Buttons
        setupBottomButtons()
        
        // Setup TableView
        setupTableView()
        
        
    }
    
    func setupTableView() {
        view.addSubview(tableView)
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        
        tableView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: buttonContainer.topAnchor)
        ])
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
        button2.setTitle("CSV", for: .normal)
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
                presentingVC.repoDictionary = self.repoDictionary
                dismiss(animated: true, completion: nil) // Return to the existing view
        } else {
            let targetViewController = storyboard!.instantiateViewController(withIdentifier: "scanview") as! ViewController
            targetViewController.modalPresentationStyle = .fullScreen
            targetViewController.repoDictionary = self.repoDictionary
            present(targetViewController, animated: true, completion: nil)
        }
    }
    
    @objc func button2Tapped() {
        print("Button 2 tapped")
        csvEmail()
    }
    
    func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
        controller.dismiss(animated: true)
    }
    
    @objc func csvEmail() {
        if MFMailComposeViewController.canSendMail() {
            let today: Date = Date()
            let dateFormatter: DateFormatter = DateFormatter()
            dateFormatter.dateFormat = "MM-dd-yyyy HH:mm"
            let date = dateFormatter.string(from: today)
            let mail = MFMailComposeViewController()
            mail.mailComposeDelegate = self
            mail.setSubject("from Total Tracker")
            var csvString = "date,shop,total\n" // Header row
            for entry in repoDictionary {
                if let date = entry["date"], let shop = entry["shop"], let total = entry["total"] {
                    let row = "\(date),\(shop),\(total)\n"
                    csvString.append(row)
                }
            }
            data = csvString.data(using: .utf8)
            mail.addAttachmentData(data!, mimeType: "text/csv", fileName: date + ".csv")
            present(mail, animated: true, completion: nil)
        }
    }
}

extension TableViewController: UITableViewDelegate, UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return repoDictionary.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        let dicObject = repoDictionary[indexPath.row]
        // Parsing each value
        let date = dicObject["date"] as? String
        let shop = dicObject["shop"] as? String
        let total = dicObject["total"] as? String
        cell.textLabel?.text = date! + ", " + shop! + ", " + total!
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let dicObject = repoDictionary[indexPath.row]
        print("Selected row: \(indexPath.row), Item: \(dicObject)")
        tableView.deselectRow(at: indexPath, animated: true)
    }
    
    func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        if editingStyle == .delete {
            self.repoDictionary.remove(at: indexPath.row)
            UserDefaults.standard.set(repoDictionary, forKey: "repoDictionary")
            if self.repoDictionary.count == 0{
                UserDefaults.standard.removeObject(forKey: "repoDictionary")
            }
            // Delete the row from the table view
            tableView.deleteRows(at: [indexPath], with: .automatic)
        }
    }
}
