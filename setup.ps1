# Create virtual environment
py -3.13 -m venv venv

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Upgrade pip
py -3.13 -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

Write-Host "Setup complete. Virtual environment created and dependencies installed."
Write-Host "To activate the virtual environment, run: .\venv\Scripts\Activate.ps1"
