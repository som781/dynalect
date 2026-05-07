# Dynalect

A prototype e-commerce storefront where the user navigates and shops by **typing natural-language commands** instead of clicking. Built in 2023 as an exploration of using a zero-shot language model as the primary UI control surface.

> **Status:** archived prototype (2023). Kept as-is for reference.

---

## What it does

You type things like:

- *"show me the cart"*
- *"sort products from cheap to expensive"*
- *"add the red running shoes to my cart"*
- *"tell me more about the leather wallet"*

…and the app routes you to the right page or performs the right action. No buttons required.

## How it works

```
   user text
       │
       ▼
┌──────────────────────┐
│  BART-large-MNLI     │   zero-shot classification against
│  (zero-shot)         │   9 candidate intent labels
└──────────┬───────────┘
           │  intent (e.g. "add to cart")
           ▼
┌──────────────────────┐
│  intent → handler    │   FastAPI route mapping
└──────────┬───────────┘
           │
           ├── for product-targeted intents:
           │     fuzzy-match (Levenshtein, fuzzywuzzy)
           │     instruction text against product names
           │     → resolve product_id
           │
           ▼
   redirect / DB action / rendered page
```

Two pieces of NLP, both running locally:

1. **Intent classification.** `facebook/bart-large-mnli` via Hugging Face `transformers` `zero-shot-classification` pipeline. The candidate labels are the 9 actions the storefront supports (`home`, `products`, `cart`, `add to cart`, `remove from cart`, `product's price high to low`, etc.). No training data, no fine-tuning — the model picks the most entailed label.
2. **Entity extraction.** For intents that target a specific product (`add to cart`, `remove from cart`, `product's details information`), the instruction is fuzzy-matched against every product name in the database using `fuzz.partial_ratio`. If the best match clears a similarity threshold (70), that product's ID is used.

## Why these choices

- **Zero-shot over fine-tuning** — there was no labeled dataset and the action set was small and well-defined. NLI-based zero-shot classification handles this regime cleanly, and changing the action set means editing a Python list, not retraining.
- **Fuzzy match over NER** — product catalogs change constantly and a NER model would need re-training each time. Levenshtein against the live `products` table always reflects current inventory.
- **Local inference** — kept the whole stack runnable on a laptop with no API keys or per-request cost.

## Tech stack

- **Backend:** FastAPI, Uvicorn
- **NLP:** Hugging Face Transformers, `facebook/bart-large-mnli`, fuzzywuzzy
- **Database:** MySQL (products, cart)
- **Frontend:** Jinja2 templates, vanilla HTML/CSS

## Project layout

```
app.py              FastAPI app — routes, DB access, intent handling
templates/          Jinja2 templates (home, products, product_detail, cart, account, popup)
static/images/      Product images
schema.rar          MySQL schema (products, cart tables)
requirements.txt    Pinned Python dependencies
test_intent.py      Scratch script for trying out the classification pipeline
```

## Running locally

```bash
# 1. MySQL
#    create database `webgpt` and load the schema from schema.rar
#    populate the `products` table with rows (id, name, price, description)

# 2. Set the DB password
export DB_PASSWORD=your_mysql_password

# 3. Install + run
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:8000
```

The first request triggers Hugging Face to download the BART-MNLI weights (~1.6 GB), so cold start is slow.

## Known limitations

This was a prototype, not a product. Things it doesn't do:

- Auth (the `/account` page is a stub).
- Quantity-aware add/remove (always ±1).
- Multi-product instructions ("add the wallet and the shoes").
- Anything resembling input validation or rate limiting on the LLM endpoint.

## Demo

Walkthrough video: https://www.youtube.com/watch?v=ab88SojP580
