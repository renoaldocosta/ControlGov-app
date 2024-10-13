# C:\SCRIPTS_INFNET\RD5_Projeto\Livro\Teste\database.py

import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId


# Conexão com o MongoDB Atlas
MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb+srv://renoaldo_teste:zfUsBhhQFvz5hnEV@cluster0.zmdkz1p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_DETAILS)

database = client.livro  # Nome do banco de dados

# Coleções
product_collection = database.get_collection("products")

# Função para converter documentos do MongoDB para dicionário com id como string
def product_helper(product) -> dict:
    return {
        "id": str(product["_id"]),
        "name": product["name"],
        "description": product.get("description", ""),
        "price": product["price"],
        "quantity": product["quantity"],
    }
