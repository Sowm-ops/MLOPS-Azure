pipeline {
    agent any

    environment {
        AZURE_STORAGE_ACCOUNT = credentials('AZURE_STORAGE_ACCOUNT_JENKINS')
        AZURE_STORAGE_KEY     = credentials('AZURE_STORAGE_KEY_JENKINS')

        PIP_CACHE_DIR = "${WORKSPACE}\\.pip-cache"
        VENV_DIR      = "${WORKSPACE}\\.venv"
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
                pip install dvc[azure] pytest
                '''
            }
        }

        stage('Configure DVC Remote (Jenkins only)') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"

                dvc remote add -f azurejenkins azure://dvc-jenkins/
                dvc remote modify --local azurejenkins account_name "%AZURE_STORAGE_ACCOUNT%"
                dvc remote modify --local azurejenkins account_key "%AZURE_STORAGE_KEY%"
                dvc remote default azurejenkins

                dvc remote list
                '''
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

        stage('DVC Pull') {
            steps {
                bat '''
                call "%VENV_DIR%\\Scripts\\activate"
                dvc pull -v || echo "No cache yet"
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
                dvc push -v
                '''
            }
        }

        stage('Archive Metrics') {
            steps {
                archiveArtifacts artifacts: 'metrics/**', allowEmptyArchive: true
            }
        }
    }
}
