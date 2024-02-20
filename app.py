# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo
import os

# Importa la biblioteca Streamlit para crear aplicaciones web interactivas
import streamlit as st
# Importa la biblioteca xml.etree.ElementTree para procesar archivos XML
import xml.etree.ElementTree as ET

# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks.tracers import langchain
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain.llms import OpenAI
# Importa CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter

# Importa SQLite para manejar la base de datos
import sqlite3

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para inicializar la base de datos SQLite
def initialize_database():
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS text_fragments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, fragment TEXT)''')
    conn.commit()
    conn.close()

# Función para procesar el texto extraído de un archivo
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Inserta los trozos de texto en la base de datos SQLite
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    for chunk in chunks:
        c.execute("INSERT INTO text_fragments (fragment) VALUES (?)", (chunk,))
    conn.commit()
    conn.close()

# Función principal de la aplicación
def main():
    st.title("Preguntas a un Archivo")  # Establece el título de la aplicación

    files = st.file_uploader("Sube tus archivos (máximo 4)", type=["xml"],
                             accept_multiple_files=True)  # Crea un cargador de archivos para subir archivos XML

    if files is not None and len(files) <= 4:
        total_tokens_consumed = 0

        # Inicializa la base de datos SQLite
        initialize_database()

        for file in files:
            # Procesar archivos XML
            xml_root = ET.parse(file).getroot()
            text = " ".join([elem.text for elem in xml_root.iter() if elem.text])

            process_text(text)
            total_tokens_consumed += len(text.split())

        st.write(f"Total de tokens consumidos: {total_tokens_consumed}")

        # Caja de entrada de texto para que el usuario escriba su pregunta
        query = st.text_input('Escribe tu pregunta para el archivo...')

        # Botón para cancelar la pregunta
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()  # Detiene la ejecución de la aplicación

        if query:
            # Realiza una búsqueda en la base de datos SQLite
            conn = sqlite3.connect('knowledge_base.db')
            c = conn.cursor()
            c.execute("SELECT fragment FROM text_fragments WHERE fragment LIKE ?", ('%' + query + '%',))
            results = c.fetchall()
            docs = [result[0] for result in results]
            conn.close()

            # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
            model = "gpt-3.5-turbo-instruct"  # Acepta 4096 tokens
            temperature = 0  # Valores entre 0 - 1
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

            # Carga la cadena de preguntas y respuestas
            chain = load_qa_chain(llm, chain_type="stuff")

            # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                print(cost)  # Imprime el costo de la operación

                st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación

# Punto de entrada para la ejecución del programa
if __name__== "__main__":
    main()  # Llama a la función principal
