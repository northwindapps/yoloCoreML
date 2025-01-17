//
//  Menu.swift
//  coreMLYOLO
//
//  Created by yano on 2025/01/17.
//

import UIKit

class Menu: UIView {
    
    var view:UIView!
    
    override init(frame: CGRect)
    {
        super.init(frame: frame)
        setup()
    }
    
    required init(coder aDecoder:NSCoder)
    {
        super.init(coder:aDecoder)!
        setup()
    }
    
    func setup()
    {
        view = loadviewfromNib()
        view.frame = bounds
        //http://stackoverflow.com/questions/30867325/binary-operator-cannot-be-applied-to-two-UIView.AutoresizingMask-operands
        view.autoresizingMask = [UIView.AutoresizingMask.flexibleWidth, UIView.AutoresizingMask.flexibleHeight]
        addSubview(view)
        
    }
    
    //http://stackoverflow.com/questions/34658838/instantiate-view-from-nib-throws-error
    func loadviewfromNib() ->UIView
    {
        let bundle = Bundle(for: type(of: self))
        let nib = UINib(nibName: "menu",bundle: bundle)
        let view = nib.instantiate(withOwner: self, options: nil)[0] as! UIView
        
        return view
    }
    
}
