import uuid
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Abstract base class for toxic content
class ToxicContent(ABC):
    def __init__(self, content: str, severity: float):
        if not (0.0 <= severity <= 1.0):
            raise ValueError("Severity must be between 0.0 and 1.0")
        self.id = str(uuid.uuid4())
        self.content = content
        self.severity = severity
        self.timestamp = datetime.now()
        self.category = self._get_category()

    @abstractmethod
    def _get_category(self) -> str:
        pass

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category
        }

    def __str__(self) -> str:
        return f"[{self.category}] Severity: {self.severity:.2f} - {self.content}"

# Concrete classes for specific toxic content types
class ToxicSportsContent(ToxicContent):
    def _get_category(self) -> str:
        return "Toxic Sports Content"

class ToxicEconomicContent(ToxicContent):
    def _get_category(self) -> str:
        return "Toxic Economic Content"

class ToxicPoliticalContent(ToxicContent):
    def _get_category(self) -> str:
        return "Toxic Political Content"

# Manager class to handle toxic content
class ToxicContentManager:
    def __init__(self):
        self.contents: List[ToxicContent] = []
        self._observers = []

    def add_content(self, content: ToxicContent) -> None:
        self.contents.append(content)
        logger.info(f"Added content: {content}")
        self._notify_observers()

    def remove_content(self, content_id: str) -> bool:
        initial_length = len(self.contents)
        self.contents = [c for c in self.contents if c.id != content_id]
        if len(self.contents) < initial_length:
            logger.info(f"Removed content with ID: {content_id}")
            self._notify_observers()
            return True
        logger.warning(f"Content with ID {content_id} not found")
        return False

    def get_content_by_id(self, content_id: str) -> Optional[ToxicContent]:
        for content in self.contents:
            if content.id == content_id:
                return content
        logger.warning(f"Content with ID {content_id} not found")
        return None

    def filter_by_category(self, category: str) -> List[ToxicContent]:
        return [c for c in self.contents if c.category == category]

    def filter_by_severity(self, min_severity: float, max_severity: float = 1.0) -> List[ToxicContent]:
        return [c for c in self.contents if min_severity <= c.severity <= max_severity]

    def sort_by_severity(self, reverse: bool = False) -> List[ToxicContent]:
        return sorted(self.contents, key=lambda x: x.severity, reverse=reverse)

    def sort_by_timestamp(self, reverse: bool = False) -> List[ToxicContent]:
        return sorted(self.contents, key=lambda x: x.timestamp, reverse=reverse)

    def register_observer(self, observer) -> None:
        self._observers.append(observer)
        logger.info("Registered new observer")

    def _notify_observers(self) -> None:
        for observer in self._observers:
            observer.update(self.contents)

    def serialize_to_json(self, filename: str) -> None:
        try:
            with open(filename, 'w') as f:
                json.dump([c.to_dict() for c in self.contents], f, indent=2)
            logger.info(f"Serialized content to {filename}")
        except Exception as e:
            logger.error(f"Failed to serialize content: {str(e)}")

    def deserialize_from_json(self, filename: str) -> None:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.contents.clear()
            for item in data:
                category = item.get("category")
                content = item.get("content")
                severity = item.get("severity")
                if category == "Toxic Sports Content":
                    new_content = ToxicSportsContent(content, severity)
                elif category == "Toxic Economic Content":
                    new_content = ToxicEconomicContent(content, severity)
                elif category == "Toxic Political Content":
                    new_content = ToxicPoliticalContent(content, severity)
                else:
                    continue
                new_content.id = item.get("id")
                new_content.timestamp = datetime.fromisoformat(item.get("timestamp"))
                self.contents.append(new_content)
            logger.info(f"Deserialized content from {filename}")
            self._notify_observers()
        except Exception as e:
            logger.error(f"Failed to deserialize content: {str(e)}")

# Observer interface for updates
class ContentObserver(ABC):
    @abstractmethod
    def update(self, contents: List[ToxicContent]) -> None:
        pass

# Concrete observer for monitoring high-severity content
class HighSeverityObserver(ContentObserver):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def update(self, contents: List[ToxicContent]) -> None:
        high_severity = [c for c in contents if c.severity >= self.threshold]
        if high_severity:
            logger.warning(f"High severity content detected: {len(high_severity)} items")
            for content in high_severity:
                logger.warning(str(content))

# Analysis class for generating reports
class ContentAnalyzer:
    def __init__(self, manager: ToxicContentManager):
        self.manager = manager

    def generate_category_report(self) -> dict:
        report = {}
        for content in self.manager.contents:
            report[content.category] = report.get(content.category, 0) + 1
        return report

    def generate_severity_distribution(self, bins: int = 5) -> dict:
        distribution = {f"{i/bins:.2f}-{(i+1)/bins:.2f}": 0 for i in range(bins)}
        for content in self.manager.contents:
            bin_key = f"{int(content.severity * bins) / bins:.2f}-{(int(content.severity * bins) + 1) / bins:.2f}"
            distribution[bin_key] = distribution.get(bin_key, 0) + 1
        return distribution

# Command-line interface for interaction
class ContentCLI:
    def __init__(self, manager: ToxicContentManager, analyzer: ContentAnalyzer):
        self.manager = manager
        self.analyzer = analyzer

    def add_content_interactive(self, category: str, content: str, severity: float) -> None:
        try:
            if category.lower() == "sports":
                new_content = ToxicSportsContent(content, severity)
            elif category.lower() == "economic":
                new_content = ToxicEconomicContent(content, severity)
            elif category.lower() == "political":
                new_content = ToxicPoliticalContent(content, severity)
            else:
                logger.error("Invalid category")
                return
            self.manager.add_content(new_content)
        except ValueError as e:
            logger.error(f"Error adding content: {str(e)}")

    def display_all(self) -> None:
        for content in self.manager.contents:
            print(content)

    def display_report(self) -> None:
        print("\nCategory Report:")
        for category, count in self.analyzer.generate_category_report().items():
            print(f"{category}: {count} items")
        print("\nSeverity Distribution:")
        for bin_range, count in self.analyzer.generate_severity_distribution().items():
            print(f"Severity {bin_range}: {count} items")

# Example usage
def main():
    # Initialize components
    manager = ToxicContentManager()
    analyzer = ContentAnalyzer(manager)
    cli = ContentCLI(manager, analyzer)

    # Register observer
    observer = HighSeverityObserver(threshold=0.8)
    manager.register_observer(observer)

    # Add sample content
    cli.add_content_interactive("sports", "Violent fan behavior encouragement", 0.9)
    cli.add_content_interactive("economic", "False stock market claims", 0.7)
    cli.add_content_interactive("political", "Polarization incitement", 0.85)
    cli.add_content_interactive("sports", "Mild toxic sports comment", 0.4)

    # Display all content
    print("\nAll Toxic Content:")
    cli.display_all()

    # Display reports
    cli.display_report()

    # Filter and sort example
    print("\nHigh Severity Content (Severity >= 0.8):")
    for content in manager.filter_by_severity(0.8):
        print(content)

    print("\nSorted by Severity (Descending):")
    for content in manager.sort_by_severity(reverse=True):
        print(content)

    # Serialize to JSON
    manager.serialize_to_json("toxic_content.json")

if __name__ == "__main__":
    main()