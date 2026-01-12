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
                if not exist "%VENV_DIR%" python -m venv "%VENV_DIR%"
                call "%VENV_DIR%\\Scripts\\activate"

                python -m pip install --upgrade pip
                pip config set global.cache-dir "%PIP_CACHE_DIR%"

                pip install -r requirements.txt
                pip install dvc[azure] pytest nltk
                '''
            }
        }

        stage('Configure DVC Remote') {
            steps {
                withCredentials([
                    string(credentialsId: 'AZURE_STORAGE_ACCOUNT_JENKINS', variable: 'AZURE_ACCOUNT'),
                    string(credentialsId: 'AZURE_STORAGE_KEY_JENKINS', variable: 'AZURE_KEY')
                ]) {
                    bat '''
                    call "%VENV_DIR%\\Scripts\\activate"

                    REM Force refresh the remote configuration
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

        stage('DVC Pull (Smart Sync)') {
            steps {
                withCredentials([
                    string(credentialsId: 'AZURE_STORAGE_ACCOUNT_JENKINS', variable: 'AZURE_ACCOUNT'),
                    string(credentialsId: 'AZURE_STORAGE_KEY_JENKINS', variable: 'AZURE_KEY')
                ]) {
                    bat '''
                    call "%VENV_DIR%\\Scripts\\activate"

                    echo Checking remote 'azurejenkins' status...

                    dvc list . --remote azurejenkins >nul 2>&1

                    IF %ERRORLEVEL% NEQ 0 (
                        echo [INFO] Detected EMPTY remote. Skipping pull...
                    ) ELSE (
                        echo [INFO] Remote data found. Syncing...
                        REM Added --force here to overwrite existing local files
                        dvc pull -r azurejenkins --force
                    )
                    '''
                }
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
                
                REM dvc repro runs the pipeline and generates the models/ folder
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
                withCredentials([
                    string(credentialsId: 'AZURE_STORAGE_ACCOUNT_JENKINS', variable: 'AZURE_ACCOUNT'),
                    string(credentialsId: 'AZURE_STORAGE_KEY_JENKINS', variable: 'AZURE_KEY')
                ]) {
                    bat '''
                    call "%VENV_DIR%\\Scripts\\activate"
                    echo Pushing new data/models to Azure...
                    dvc push
                    '''
                }
            }
        }

        stage('Archive Metrics') {
            steps {
                archiveArtifacts artifacts: 'metrics/**', fingerprint: true, allowEmptyArchive: true
            }
        }
    }

    post {
        always {
            echo "Pipeline finished."
        }
        failure {
            echo "Pipeline failed. Check logs for DVC or Test errors."
        }
    }
}