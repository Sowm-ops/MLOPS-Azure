pipeline {
    agent any

    environment {
        VENV_DIR = "${WORKSPACE}\\.venv"
        PIP_CACHE_DIR = "${WORKSPACE}\\.pip-cache"
        PYTHONUTF8 = "1"

        // Your Azure + ACR + App names (hardcode for stability)
        ACR_LOGIN_SERVER = "mlopsacr123.azurecr.io"
        CONTAINER_APP_NAME = "mlops-serving-jenkins"
        RESOURCE_GROUP = "mlops-rg"
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
                pip install dvc[azure] pytest nltk evidently==0.4.32
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

        // -------------------------------
        // NEW: Build & Push image to ACR
        // -------------------------------
        stage('Build & Push Image to ACR') {
            steps {
                withCredentials([
                    string(credentialsId: 'ACR_USERNAME', variable: 'ACR_USERNAME'),
                    string(credentialsId: 'ACR_PASSWORD', variable: 'ACR_PASSWORD')
                ]) {
                    bat '''
                    echo Checking docker...
                    docker version

                    set IMAGE_NAME=%ACR_LOGIN_SERVER%/mlops-api:jenkins-%BUILD_NUMBER%
                    echo Using IMAGE_NAME=%IMAGE_NAME%

                    echo %ACR_PASSWORD% | docker login %ACR_LOGIN_SERVER% -u %ACR_USERNAME% --password-stdin
                    docker build -t %IMAGE_NAME% .
                    docker push %IMAGE_NAME%
                    '''
                }
            }
        }

        // -------------------------------
        // NEW: Deploy only to Jenkins ACA
        // -------------------------------
        stage('Deploy to Azure Container App (Jenkins)') {
            steps {
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID', variable: 'AZURE_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET', variable: 'AZURE_CLIENT_SECRET'),
                    string(credentialsId: 'AZURE_TENANT_ID', variable: 'AZURE_TENANT_ID'),
                    string(credentialsId: 'AZURE_SUBSCRIPTION_ID', variable: 'AZURE_SUBSCRIPTION_ID')
                ]) {
                    bat '''
                    echo Checking az...
                    az version

                    az login --service-principal -u %AZURE_CLIENT_ID% -p %AZURE_CLIENT_SECRET% --tenant %AZURE_TENANT_ID%
                    az account set --subscription %AZURE_SUBSCRIPTION_ID%

                    set IMAGE_NAME=%ACR_LOGIN_SERVER%/mlops-api:jenkins-%BUILD_NUMBER%
                    echo Deploying IMAGE_NAME=%IMAGE_NAME%
                    az extension add -n containerapp --upgrade

                    az containerapp update --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --image %IMAGE_NAME%
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
            echo "Pipeline failed. Check logs for DVC, Docker, or Azure CLI errors."
        }
    }
}
