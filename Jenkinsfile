pipeline {
    agent any

    environment {
        VENV_DIR = "${WORKSPACE}\\.venv"
        PIP_CACHE_DIR = "${WORKSPACE}\\.pip-cache"
        PYTHONUTF8 = "1"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                bat '''
                python -m venv "%VENV_DIR%"
                call "%VENV_DIR%\\Scripts\\activate"

                python -m pip install --upgrade pip
                pip config set global.cache-dir "%PIP_CACHE_DIR%"

                pip install -r requirements.txt
                pip install dvc[azure] pytest nltk
                '''
            }
        }

        stage('Configure DVC Remote (Jenkins only)') {
            steps {
                withCredentials([
                    string(credentialsId: 'AZURE_STORAGE_ACCOUNT_JENKINS', variable: 'AZURE_ACCOUNT'),
                    string(credentialsId: 'AZURE_STORAGE_KEY_JENKINS', variable: 'AZURE_KEY')
                ]) {
                    bat '''
                    call "%VENV_DIR%\\Scripts\\activate"

                    dvc remote remove azurejenkins || echo "No existing Jenkins remote"
                    dvc remote add -f azurejenkins azure://dvc-jenkins
                    dvc remote modify --local azurejenkins account_name "%AZURE_ACCOUNT%"
                    dvc remote modify --local azurejenkins account_key "%AZURE_KEY%"
                    dvc remote default azurejenkins

                    dvc remote list
                    '''
                }
            }
        }

        stage('Download NLTK Data') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
                '''
            }
        }

        stage('DVC Pull (First-Run Logic)') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"

                echo Checking remote 'azurejenkins' status...

                REM 1. Check if the remote is configured
                dvc remote list | findstr "azurejenkins" >nul
                IF %ERRORLEVEL% NEQ 0 (
                    echo ERROR: Remote 'azurejenkins' not found in config!
                    exit /b 1
                )

                REM 2. Check if the remote has ANY data (the /files/md5 folder)
                REM We try to list the remote. If it's totally empty, dvc list returns error 1.
                dvc list . --remote azurejenkins >nul 2>&1

                IF %ERRORLEVEL% NEQ 0 (
                    echo "Detected EMPTY remote. This must be the first run. Skipping pull..."
                ) ELSE (
                    echo "Remote has data. Attempting sync..."
                    REM If this fails now, the whole stage fails (exit code 1)
                    dvc pull -r azurejenkins
                )
                '''
            }
        }

        stage('Pre-Training Tests') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                set PYTHONPATH=%WORKSPACE%
                pytest tests/pre -q
                '''
            }
        }

        stage('Run Training Pipeline') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                set PYTHONPATH=%WORKSPACE%

                dvc repro
                '''
            }
        }

        stage('Post-Training Tests') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                set PYTHONPATH=%WORKSPACE%
                pytest tests/post -q
                '''
            }
        }

        stage('DVC Push') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                dvc push
                '''
            }
        }

        stage('Archive Metrics') {
            steps {
                archiveArtifacts artifacts: 'metrics/**', fingerprint: true, allowEmptyArchive: true
            }
        }
    }
}
