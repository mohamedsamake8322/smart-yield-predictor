import json
import bcrypt
import os

# Informations de l'utilisateur
username = "mohamedsamake2000"
password = "78772652Moh#"
role = "admin"

# Hachage du mot de passe
hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Création du dictionnaire utilisateur
user_data = {
    username: {
        "password": hashed_pw,
        "role": role
    }
}

# Chemin vers le fichier users.json
file_path = "users.json"

# Écriture ou mise à jour du fichier users.json
if os.path.exists(file_path):
    with open(file_path, "r") as f:
        existing_data = json.load(f)
else:
    existing_data = {}

existing_data.update(user_data)

with open(file_path, "w") as f:
    json.dump(existing_data, f, indent=4)

print(f"✅ Utilisateur {username} ajouté avec succès.")
