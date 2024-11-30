import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel

def main():
    print("Starting test application...")
    app = QApplication(sys.argv)
    print("Created QApplication")
    
    window = QMainWindow()
    window.setWindowTitle("Test Window")
    window.setGeometry(100, 100, 400, 300)
    
    label = QLabel("Hello, PyQt!", parent=window)
    label.setGeometry(100, 100, 200, 50)
    
    print("Created window")
    window.show()
    print("Window should be visible now")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    print("Script is running")
    main() 