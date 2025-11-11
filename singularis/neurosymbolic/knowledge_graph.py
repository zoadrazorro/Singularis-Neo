"""Knowledge Graph for structured knowledge"""
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class Entity:
    id: str
    type: str
    properties: Dict

@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    properties: Dict = None

class KnowledgeGraph:
    """Simple in-memory knowledge graph"""
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        self.relations.append(relation)

    def query(self, subject: str = None, predicate: str = None, object: str = None) -> List[Relation]:
        results = []
        for rel in self.relations:
            if (subject is None or rel.subject == subject) and \
               (predicate is None or rel.predicate == predicate) and \
               (object is None or rel.object == object):
                results.append(rel)
        return results

    def get_neighbors(self, entity_id: str) -> Set[str]:
        neighbors = set()
        for rel in self.relations:
            if rel.subject == entity_id:
                neighbors.add(rel.object)
            elif rel.object == entity_id:
                neighbors.add(rel.subject)
        return neighbors
