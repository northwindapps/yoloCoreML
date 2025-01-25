import UIKit
import MessageUI

class EmailViewController: UIViewController,UITextFieldDelegate {
    
    var capturedImage = UIImage()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
    
    override func viewDidAppear(_ animated: Bool) {
        let appd : AppDelegate = UIApplication.shared.delegate as! AppDelegate
        super.viewDidAppear(animated)
        
        
    }
    
    
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder() // Dismiss the keyboard
        return true
    }
    
}
    
   
