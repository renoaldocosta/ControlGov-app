# C:\SCRIPTS_INFNET\RD5_Projeto\Livro\Teste\routers\product.py

from fastapi import APIRouter, HTTPException
from typing import List
from Teste.database import product_collection, product_helper
from Teste.models import Product, UpdateProduct
from bson import ObjectId

router = APIRouter()

@router.post("/", response_description="Adicionar um novo produto", response_model=dict)
async def create_product(product: Product):
    product = await product_collection.insert_one(product.dict())
    new_product = await product_collection.find_one({"_id": product.inserted_id})
    return product_helper(new_product)

@router.get("/", response_description="Lista de produtos", response_model=List[dict])
async def list_products():
    products = []
    async for product in product_collection.find():
        products.append(product_helper(product))
    return products

@router.get("/{id}", response_description="Obter um único produto", response_model=dict)
async def show_product(id: str):
    if (product := await product_collection.find_one({"_id": ObjectId(id)})) is not None:
        return product_helper(product)
    raise HTTPException(status_code=404, detail=f"Produto {id} não encontrado")

@router.put("/{id}", response_description="Atualizar um produto", response_model=dict)
async def update_product(id: str, product: UpdateProduct):
    update_data = {k: v for k, v in product.dict().items() if v is not None}

    if len(update_data) >= 1:
        update_result = await product_collection.update_one({"_id": ObjectId(id)}, {"$set": update_data})

        if update_result.modified_count == 1:
            if (updated_product := await product_collection.find_one({"_id": ObjectId(id)})) is not None:
                return product_helper(updated_product)

    if (existing_product := await product_collection.find_one({"_id": ObjectId(id)})) is not None:
        return product_helper(existing_product)

    raise HTTPException(status_code=404, detail=f"Produto {id} não encontrado")

@router.delete("/{id}", response_description="Remover um produto")
async def delete_product(id: str):
    delete_result = await product_collection.delete_one({"_id": ObjectId(id)})

    if delete_result.deleted_count == 1:
        return {"message": f"Produto {id} removido com sucesso"}

    raise HTTPException(status_code=404, detail=f"Produto {id} não encontrado")
