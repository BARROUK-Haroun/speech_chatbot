import streamlit as st
import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('wordnet')

# Chargement du fichier de données pour le chatbot
try:
    with open("chatbot_data.txt", "r", encoding="utf8") as f:
        raw_data = f.read().lower()
except FileNotFoundError:
    st.error("Le fichier 'chatbot_data.txt' est introuvable.")
    raw_data = "Bonjour, comment puis-je vous aider ? Je suis un chatbot basique."

# Prétraitement du texte
sent_tokens = sent_tokenize(raw_data)
word_tokens = word_tokenize(raw_data)

lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in '!()-[]{};:\'\"\,<>./?@#$%^&*_~')
def LemNormalize(text):
    return LemTokens(word_tokenize(text.lower().translate(remove_punct_dict)))

# Fonction de génération de réponse du chatbot (TF-IDF + similarité cosinus)
def generate_response(user_response):
    response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]  # récupère l’indice de la deuxième phrase la plus similaire
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        response = "Désolé, je n'ai pas compris."
    else:
        response = sent_tokens[idx]
    sent_tokens.pop()  # retirer le message utilisateur ajouté
    return response

# Fonction de reconnaissance vocale
def transcribe_speech(selected_api="Google", language="fr-FR"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parlez maintenant...")
        try:
            audio_text = r.listen(source, timeout=5)
            st.info("Transcription en cours...")
            if selected_api == "Google":
                text = r.recognize_google(audio_text, language=language)
            elif selected_api == "Sphinx":
                text = r.recognize_sphinx(audio_text)
            else:
                text = "API non supportée."
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris votre parole."
        except sr.RequestError:
            return "Erreur lors de la requête vers l'API de reconnaissance vocale."
        except Exception as e:
            return f"Erreur : {str(e)}"

# Application principale Streamlit
def main():
    st.title("Chatbot Vocal")
    st.write("Interagissez avec le chatbot en utilisant du texte ou de la voix.")

    # Choix du mode d'entrée
    input_mode = st.radio("Choisissez votre mode d'entrée :", ("Texte", "Voix"))

    user_input = ""
    if input_mode == "Texte":
        user_input = st.text_input("Entrez votre message :", "")
    else:
        # Sélection de l'API de reconnaissance vocale
        selected_api = st.selectbox("Choisissez l'API de reconnaissance vocale :", ["Google", "Sphinx"])
        # Sélection de la langue utilisée
        language = st.text_input("Entrez le code langue (ex : 'fr-FR') :", "fr-FR")
        
        # Gestion de la pause/reprise avec le session_state
        if "paused" not in st.session_state:
            st.session_state.paused = False

        if st.button("Pause/Resume"):
            st.session_state.paused = not st.session_state.paused
            if st.session_state.paused:
                st.info("La reconnaissance vocale est en pause.")
            else:
                st.info("La reconnaissance vocale est reprise.")
        
        if not st.session_state.paused:
            if st.button("Démarrer l'enregistrement"):
                user_input = transcribe_speech(selected_api, language)
                st.write("Transcription :", user_input)

    # Si une saisie (texte ou vocale) est fournie, générer une réponse
    if user_input and user_input.strip() != "":
        response = generate_response(user_input.lower())
        st.markdown("**Chatbot :** " + response)
        
        # Option de sauvegarde de la transcription (pour l'entrée vocale)
        if input_mode == "Voix":
            if st.button("Sauvegarder la transcription"):
                with open("transcription.txt", "w", encoding="utf-8") as f:
                    f.write(user_input)
                st.success("Transcription sauvegardée dans 'transcription.txt'")
            st.download_button("Télécharger la transcription", user_input, file_name="transcription.txt", mime="text/plain")

if __name__ == "__main__":
    main()