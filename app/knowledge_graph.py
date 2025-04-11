import re
from collections import defaultdict
from neo4j import GraphDatabase
from tqdm import tqdm
import os

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class LegalKGNeo4j:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def create_article(self, law_id, number, text):
        with self.driver.session() as session:
            session.run("""
                MATCH (l:Law {id: $law_id})
                MERGE (a:Article {number: $number})
                SET a.text = $text
                MERGE (l)-[:CONTAINS]->(a)
            """, law_id=law_id, number=number, text=text)

    def create_paragraph(self, law_id, number, text):
        with self.driver.session() as session:
            session.run("""
                MATCH (l:Law {id: $law_id})
                MERGE (p:Paragraph {number: $number})
                SET p.text = $text
                MERGE (l)-[:CONTAINS]->(p)
            """, law_id=law_id, number=number, text=text)

    def create_base_nodes_and_relationships(self, law_title, parsed_text_items):
        with self.driver.session() as session:
            session.run("MERGE (:Law {id: $law_id, title: $law_title})",
                        law_id="BO-Wien", law_title=law_title)
            
            for index, article in enumerate(parsed_text_items["articles"]):
                # Add article as instance and its text as description
                self.create_article("BO-Wien", index, article)

            for index, paragraph in enumerate(parsed_text_items["paragraphs"]):
                # Add article as instance and its text as description
                self.create_paragraph("BO-Wien", index, paragraph)



def parse_legal_document(text: str):

    parsed_items = defaultdict()
    # Parse all articles
    articles = re.split(r'Art.\s[0-9]\nText\n+ARTIKEL\s', text)

    #TODO You could create a test here to ensure that length of 7 must be true
    print(len(articles))

    # Parse all paragraphs
    paragraphs = re.split(r'§\s\d+[a-z]?\nText\n', articles[-1])
    print(len(paragraphs))

    # Remove final article item with paragraphs
    articles.pop()
    articles.append(paragraphs[0])
    paragraphs.pop(0)

    parsed_items["articles"] = articles
    parsed_items["paragraphs"] = paragraphs

    return parsed_items



def run_query(cypher: str):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        results = session.run(cypher)
        rows = results.data()
        if not rows:
            print("Keine passenden Bestimmungen gefunden.")
        else:
            for r in rows:
                print(f"Artikel {r['article']}, Abschnitt {r['section']}:\n")
                
    driver.close()


if __name__ == "__main__":
    with open("../data/legal-basis.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Parsing document...")
    parsed_articles = parse_legal_document(raw_text)

    print("Connecting to Neo4j...")
    kg = LegalKGNeo4j()
    print("Populating Knowledge Graph...")

    kg.create_base_nodes_and_relationships("Bauordnung für Wien", parsed_articles)
    print("Knowledge Graph population complete...")
    kg.close()
