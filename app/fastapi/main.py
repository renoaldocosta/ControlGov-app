# C:\SCRIPTS_INFNET\RD5_Projeto\Livro\Teste\main.py

from fastapi import FastAPI
from Teste.routers import product


app = FastAPI()

app.include_router(product.router, prefix="/products", tags=["products"])

@app.get("/")
def read_root():
    return {"message": "Bem-vindo ao CRUD de Produtos com FastAPI e MongoDB Atlas"}


