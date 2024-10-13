# C:\SCRIPTS_INFNET\RD5_Projeto\Livro\Teste\models.py

from pydantic import BaseModel, Field
from typing import Optional

class Product(BaseModel):
    name: str = Field(..., examples=["Livro Python"])
    description: Optional[str] = Field(None, examples=["Um livro sobre programação em Python."])
    price: float = Field(..., examples=[29.99])
    quantity: int = Field(..., examples=[100])

class UpdateProduct(BaseModel):
    name: Optional[str] = Field(None, examples=["Livro Python Avançado"])
    description: Optional[str] = Field(None, examples=["Uma versão avançada do livro de Python."])
    price: Optional[float] = Field(None, examples=[39.99])
    quantity: Optional[int] = Field(None, examples=[150])
