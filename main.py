from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from typing import List
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangChainFAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Init FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COCKTAIL_SYSTEM_PROMPT = """You are a knowledgeable bartender assistant with access to a cocktail database and user preferences. 
You should help users find cocktails, learn about ingredients, and manage their preferences.

When discussing cocktails:
1. If asked about specific cocktails or ingredients, use the provided cocktail database information
2. If user preferences are available, consider them in your recommendations
3. Be helpful and informative about both alcoholic and non-alcoholic cocktails
4. If no specific cocktail information is provided, you can still give general beverage advice

Remember: You have access to the complete cocktail database and user preferences in the context."""


class CocktailAdvisor:
    def __init__(self):
        self.df = pd.read_csv("datasets/final_cocktails.csv")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize LLM with system prompt
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7
        )

        # Setup conversation memory with custom prompt
        self.memory = ConversationBufferMemory()
        cocktail_list = "\n".join(
            [f"{row['name']}: {row['ingredients']}" for _,
                row in self.df.iterrows()]
        )
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template=f"{COCKTAIL_SYSTEM_PROMPT}\n\nCocktail List:\n{cocktail_list}\n\nChat History:\n{{history}}\nHuman: {{input}}\nAssistant:"
        )

        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt_template,
            verbose=True
        )

        # Initialize vector stores
        self.setup_vector_stores()

    def setup_vector_stores(self):
        # Convert cocktail data to documents
        cocktail_docs = [
            Document(
                page_content=f"Cocktail: {row['name']}\nIngredients: {row['ingredients']}",
                metadata={"name": row["name"],
                          "ingredients": row["ingredients"]}
            )
            for _, row in self.df.iterrows()
        ]

        # Setup cocktail vector store
        self.cocktail_vectorstore = LangChainFAISS.from_documents(
            cocktail_docs,
            self.embeddings
        )

        # Setup user preferences vector store
        if os.path.exists("indexes/user_preferences.faiss"):
            try:
                self.user_vectorstore = LangChainFAISS.load_local(
                    "indexes/user_preferences",
                    self.embeddings
                )
            except Exception:
                self.user_vectorstore = self._create_empty_vectorstore()
        else:
            self.user_vectorstore = self._create_empty_vectorstore()

    def _create_empty_vectorstore(self):
        dummy_doc = Document(
            page_content="initialization document",
            metadata={"type": "dummy"}
        )
        vectorstore = LangChainFAISS.from_documents(
            [dummy_doc],
            self.embeddings
        )
        return vectorstore

    def save_user_preferences(self, preferences: List[str]):
        if not preferences:
            return

        docs = [Document(page_content=pref) for pref in preferences]

        # Remove dummy document if it exists
        if (len(self.user_vectorstore.index_to_docstore_id) == 1 and
                next(iter(self.user_vectorstore.docstore._dict.values())).metadata.get("type") == "dummy"):
            self.user_vectorstore = LangChainFAISS.from_documents(
                docs,
                self.embeddings
            )
        else:
            self.user_vectorstore.add_documents(docs)

        self.user_vectorstore.save_local("indexes/user_preferences")

    def get_user_preferences(self) -> List[str]:
        if len(self.user_vectorstore.index_to_docstore_id) <= 1:
            return []

        results = self.user_vectorstore.similarity_search(
            "preferred ingredients",
            k=5
        )
        return [doc.page_content for doc in results
                if doc.metadata.get("type") != "dummy"]

    def is_preference_related(self, query: str) -> bool:
        preference_keywords = [
            "favorite", "favourite", "prefer", "like", "love", "add to",
            "save", "remember", "my", "ingredients i"
        ]
        return any(keyword in query.lower() for keyword in preference_keywords)

    def search_cocktails(self, query: str) -> List[str]:
        results = self.cocktail_vectorstore.similarity_search(
            query,
            k=5
        )
        return [doc.metadata["name"] for doc in results]

    def generate_response(self, query: str) -> str:
        # Check if the query is about user preferences

        context_parts = []

        # Add user preferences if they exist
        preferences = self.get_user_preferences()
        if preferences:
            preferences_context = "Your favorite ingredients: " + \
                ", ".join(preferences)
            context_parts.append(preferences_context)

        # Combine context if any exists
        context = "\n\n".join(
            context_parts) if context_parts else "No specific cocktail information available."

        # Create prompt with context
        full_prompt = f"""Based on the following information:

{context}

Please answer this question as a knowledgeable bartender: {query}"""

        # Generate response using conversation chain
        response = self.conversation.predict(input=full_prompt)
        return response


advisor = CocktailAdvisor()

# Pydantic


class ChatRequest(BaseModel):
    message: str


class SearchRequest(BaseModel):
    query: str


class RecommendRequest(BaseModel):
    ingredients: List[str] = []


# API endpoints
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/chat/")
async def chat(request: ChatRequest):
    response = advisor.generate_response(request.message)
    return {"response": response}


@app.post("/search/")
async def search(request: SearchRequest):
    results = advisor.search_cocktails(request.query)
    return {"similar_cocktails": results}


@app.post("/recommendations/")
async def recommend(request: RecommendRequest):
    preferences = request.ingredients if request.ingredients else advisor.get_user_preferences()
    if not preferences:
        return {"recommendations": [], "message": "No preferences or ingredients provided."}

    # Search for cocktails similar to preferences
    results = []
    for pref in preferences:
        results.extend(advisor.search_cocktails(pref))

    # Remove duplicates and limit to top 5
    recommendations = list(dict.fromkeys(results))[:5]
    return {"recommendations": recommendations}


@app.get("/user/preferences/")
async def get_preferences():
    return {"preferences": advisor.get_user_preferences()}


@app.post("/user/preferences/")
async def save_preferences(request: RecommendRequest):
    advisor.save_user_preferences(request.ingredients)
    return {"message": f"Saved preferences: {', '.join(request.ingredients)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
