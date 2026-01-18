"""
Application locale pour l'analyse de sentiment avec le modèle entraîné
Utilise model_weights.pth et vocabulary.pkl téléchargés depuis Colab
"""

import torch
import torch.nn as nn
import re
import pickle
import os

# ============================================================================
# 1. REDÉFINIR LA CLASSE DU MODÈLE (identique à Colab)
# ============================================================================

class SentimentLSTM(nn.Module):
    """
    Modèle LSTM pour l'analyse de sentiment
    DOIT être identique à la définition sur Colab
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, pad_idx):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        
        return output


# ============================================================================
# 2. REDÉFINIR LE PRÉPROCESSEUR (identique à Colab)
# ============================================================================

class TextPreprocessor:
    """
    Classe pour le prétraitement du texte
    DOIT être identique à la définition sur Colab
    """
    
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
    def clean_text(self, text):
        text = re.sub(r'<.*?>', ' ', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        return text.split()
    
    def text_to_indices(self, text):
        tokens = self.tokenize(self.clean_text(text))
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]


# ============================================================================
# 3. CHARGER LE MODÈLE DEPUIS LES POIDS
# ============================================================================

def load_model(model_path='model_weights.pth'):
    """
    Charge le modèle depuis les poids sauvegardés
    
    Args:
        model_path: Chemin vers le fichier des poids
    
    Returns:
        model: Modèle chargé
        device: Device utilisé (cpu ou cuda)
    """
    # Détecter si GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Charger le checkpoint (plus sûr avec weights_only=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Récupérer les hyperparamètres
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    output_dim = checkpoint['output_dim']
    n_layers = checkpoint['n_layers']
    dropout = checkpoint['dropout']
    pad_idx = checkpoint['pad_idx']
    
    # Reconstruire le modèle avec les mêmes paramètres
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Mode évaluation
    
    print("✓ Modèle chargé avec succès!")
    print(f"  - Vocabulaire: {vocab_size} mots")
    print(f"  - Embedding: {embedding_dim}D")
    print(f"  - Hidden: {hidden_dim}D")
    print(f"  - Layers: {n_layers}")
    
    return model, device


# ============================================================================
# 4. CHARGER LE VOCABULAIRE DEPUIS LE FICHIER PICKLE
# ============================================================================

def load_preprocessor(vocab_path='vocabulary.pkl'):
    """
    Charge le vocabulaire depuis le fichier pickle
    
    Args:
        vocab_path: Chemin vers le fichier vocabulary.pkl
    
    Returns:
        preprocessor: TextPreprocessor avec vocabulaire chargé
    """
    preprocessor = TextPreprocessor()
    
    if not os.path.exists(vocab_path):
        print(f"\n⚠️ ATTENTION: '{vocab_path}' non trouvé!")
        print("Le modèle ne pourra pas faire de prédictions correctes.")
        print("\n📋 Pour télécharger le vocabulaire depuis Colab:")
        print("   1. Exécutez ce code sur Colab après l'entraînement:")
        print("      " + "-" * 50)
        print("      import pickle")
        print("      from google.colab import files")
        print("")
        print("      vocab_data = {")
        print("          'word2idx': preprocessor.word2idx,")
        print("          'idx2word': preprocessor.idx2word,")
        print("          'max_vocab_size': preprocessor.max_vocab_size")
        print("      }")
        print("")
        print("      with open('vocabulary.pkl', 'wb') as f:")
        print("          pickle.dump(vocab_data, f)")
        print("")
        print("      files.download('vocabulary.pkl')")
        print("      " + "-" * 50)
        print("   2. Placez vocabulary.pkl dans ce dossier")
        print("\nUtilisation du vocabulaire minimal (prédictions approximatives)...\n")
        
        # Créer un vocabulaire minimal
        preprocessor.word2idx = {'<PAD>': 0, '<UNK>': 1}
        preprocessor.idx2word = {0: '<PAD>', 1: '<UNK>'}
        return preprocessor
    
    try:
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        preprocessor.word2idx = vocab_data['word2idx']
        preprocessor.idx2word = vocab_data['idx2word']
        preprocessor.max_vocab_size = vocab_data['max_vocab_size']
        
        print(f"✓ Vocabulaire chargé: {len(preprocessor.word2idx):,} mots")
        return preprocessor
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du vocabulaire: {e}")
        print("Utilisation du vocabulaire minimal...")
        preprocessor.word2idx = {'<PAD>': 0, '<UNK>': 1}
        preprocessor.idx2word = {0: '<PAD>', 1: '<UNK>'}
        return preprocessor


# ============================================================================
# 5. FONCTION DE PRÉDICTION
# ============================================================================

def predict_sentiment(text, model, preprocessor, device):
    """
    Prédit le sentiment d'un texte
    
    Args:
        text: Texte à analyser
        model: Modèle chargé
        preprocessor: Préprocesseur
        device: CPU ou GPU
    
    Returns:
        sentiment: 'positive' ou 'negative'
        probability: Probabilité de la prédiction (0-1)
    """
    model.eval()
    
    # Prétraiter le texte
    indices = preprocessor.text_to_indices(text)
    
    # Si la liste est vide, retourner un résultat neutre
    if not indices:
        return 'neutral', 0.5
    
    # Convertir en tenseur
    tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    sentiment = 'positive' if prediction.item() == 1 else 'negative'
    probability = probabilities[0][prediction].item()
    
    return sentiment, probability


# ============================================================================
# 6. APPLICATION PRINCIPALE
# ============================================================================

def main():
    """
    Application principale en ligne de commande
    """
    print("=" * 60)
    print("ANALYSEUR DE SENTIMENT - IMDB")
    print("=" * 60)
    
    # Charger le modèle
    try:
        model, device = load_model('model_weights.pth')
    except FileNotFoundError:
        print("\n❌ Erreur: Fichier 'model_weights.pth' non trouvé")
        print("Assurez-vous que le fichier est dans le même dossier que ce script")
        print("\nFichiers dans le dossier actuel:")
        print([f for f in os.listdir('.') if f.endswith(('.pth', '.pkl'))])
        return
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # Charger le vocabulaire
    preprocessor = load_preprocessor('vocabulary.pkl')
    
    # Vérifier si le vocabulaire est complet
    vocab_is_complete = len(preprocessor.word2idx) > 100
    
    if not vocab_is_complete:
        print("\n" + "⚠️ " * 20)
        print("ATTENTION: Vocabulaire incomplet!")
        print("Les prédictions seront très approximatives.")
        print("Veuillez télécharger vocabulary.pkl depuis Colab.")
        print("⚠️ " * 20 + "\n")
    
    # Boucle interactive
    print("\nEntrez vos critiques de films (tapez 'quit' pour quitter):\n")
    
    while True:
        # Demander une entrée
        text = input("Votre critique: ").strip()
        
        # Quitter si demandé
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir!")
            break
        
        # Ignorer les entrées vides
        if not text:
            continue
        
        # Faire la prédiction
        sentiment, probability = predict_sentiment(text, model, preprocessor, device)
        
        # Afficher le résultat avec des emojis
        if sentiment == 'positive':
            emoji = "😊"
            color_start = "\033[92m"  # Vert
        else:
            emoji = "😞"
            color_start = "\033[91m"  # Rouge
        color_end = "\033[0m"  # Reset
        
        print(f"\n{emoji} Sentiment: {color_start}{sentiment.upper()}{color_end}")
        print(f"   Confiance: {probability * 100:.2f}%")
        
        # Avertissement si vocabulaire incomplet
        if not vocab_is_complete:
            print("   ⚠️  (Prédiction approximative - vocabulaire manquant)")
        
        print()


# ============================================================================
# 7. EXEMPLES DE TEST
# ============================================================================

def test_examples():
    """
    Teste le modèle sur quelques exemples prédéfinis
    """
    print("=" * 60)
    print("TEST SUR EXEMPLES PRÉDÉFINIS")
    print("=" * 60)
    
    # Charger le modèle
    try:
        model, device = load_model('model_weights.pth')
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # Charger le vocabulaire
    preprocessor = load_preprocessor('vocabulary.pkl')
    
    # Exemples de test
    test_reviews = [
        ("This movie was absolutely fantastic! I loved every minute of it.", "positive"),
        ("Terrible film, waste of time and money. Very disappointed.", "negative"),
        ("An okay movie, nothing special but not terrible either.", "neutral"),
        ("One of the best films I've ever seen. Incredible performances!", "positive"),
        ("Boring and predictable. I fell asleep halfway through.", "negative"),
        ("Amazing cinematography and outstanding acting!", "positive"),
        ("Worst movie ever. Do not watch.", "negative"),
        ("Good movie", "positive"),
        ("Bad movie", "negative")
    ]
    
    correct = 0
    total = 0
    
    for review, expected in test_reviews:
        sentiment, prob = predict_sentiment(review, model, preprocessor, device)
        emoji = "😊" if sentiment == 'positive' else "😞"
        
        # Vérifier si correct (pour les cas non-neutres)
        if expected != "neutral":
            is_correct = sentiment == expected
            if is_correct:
                correct += 1
            total += 1
            status = "✓" if is_correct else "✗"
        else:
            status = "~"
        
        print(f"\n{status} {emoji} {review}")
        print(f"      Prédit: {sentiment.upper()} ({prob * 100:.2f}%)")
        if expected != "neutral":
            print(f"      Attendu: {expected.upper()}")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n{'=' * 60}")
        print(f"Accuracy sur les exemples: {correct}/{total} ({accuracy:.1f}%)")
        print(f"{'=' * 60}")


# ============================================================================
# 8. FONCTION BATCH (analyser plusieurs critiques)
# ============================================================================

def analyze_batch(reviews_list):
    """
    Analyse un lot de critiques
    
    Args:
        reviews_list: Liste de textes à analyser
    """
    print("=" * 60)
    print("ANALYSE EN LOT")
    print("=" * 60)
    
    # Charger le modèle
    try:
        model, device = load_model('model_weights.pth')
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    # Charger le vocabulaire
    preprocessor = load_preprocessor('vocabulary.pkl')
    
    results = []
    
    for i, review in enumerate(reviews_list, 1):
        sentiment, prob = predict_sentiment(review, model, preprocessor, device)
        results.append({
            'review': review,
            'sentiment': sentiment,
            'probability': prob
        })
        
        emoji = "😊" if sentiment == 'positive' else "😞"
        print(f"\n{i}. {review[:60]}{'...' if len(review) > 60 else ''}")
        print(f"   {emoji} {sentiment.upper()} ({prob * 100:.2f}%)")
    
    # Statistiques
    positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
    negative_count = len(results) - positive_count
    
    print(f"\n{'=' * 60}")
    print(f"RÉSUMÉ: {positive_count} positives, {negative_count} négatives")
    print(f"{'=' * 60}")
    
    return results


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Mode interactif par défaut
    if len(sys.argv) == 1:
        main()
    
    # Mode test
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        test_examples()
    
    # Mode batch
    elif len(sys.argv) == 2 and sys.argv[1] == 'batch':
        sample_reviews = [
            "Great movie, loved it!",
            "Terrible waste of time.",
            "The best film I've seen this year.",
            "Boring and disappointing.",
            "Absolutely brilliant performances!"
        ]
        analyze_batch(sample_reviews)
    
    # Aide
    else:
        print("Usage:")
        print("  python app.py          # Mode interactif")
        print("  python app.py test     # Tester sur exemples prédéfinis")
        print("  python app.py batch    # Analyser un lot de critiques")