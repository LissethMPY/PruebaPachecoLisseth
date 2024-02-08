## INSTALACIÓN

1.- Clona el repositorio:
2.- Accede al directorio / abre con vscode
3.- Crea un Entorno Virtual:
	python -m venv env
4.- Activa el entorno virtual: source env/bin/activate - para MacOS, env/Scripts/activate - para Linux, env/Scripts/activate.bat - para Windows cmd, env/Scripts/Activate.ps1 - para Windows PowerShell
5.- Instala las dependencias requeridas:
6.- pip install -r requirements.txt
7.- Crea una Clave de API de OpenAI y agrégala a tu archivo .txt y cambia su extensión a .env
8.- Ejecuta la aplicación:
	streamlit run app.py