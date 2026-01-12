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

        stage('DVC Pull (safe)') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                dvc pull --force || echo "No cache yet (expected on first run)"
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
