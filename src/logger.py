"""
Logging configuration for Reliability-Oriented LLM Testing Framework.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

class ExperimentLogger:
    """Centralized logging system for the experiment."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Base directory for log files
        """
        self.log_dir = Path(log_dir)
        self.loggers = {}
        self._setup_directories()
        self._setup_loggers()
    
    def _setup_directories(self):
        """Create necessary log directories."""
        directories = [
            "system",
            "dataset", 
            "mutation",
            "execution",
            "evaluation",
            "metrics",
            "errors"
        ]
        
        for directory in directories:
            (self.log_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_loggers(self):
        """Setup all loggers with appropriate formatting."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # System logger
        self.loggers['system'] = self._create_logger(
            'system',
            self.log_dir / 'system' / 'system.log',
            log_format,
            date_format
        )
        
        # Dataset logger
        self.loggers['dataset'] = self._create_logger(
            'dataset',
            self.log_dir / 'dataset' / 'dataset.log',
            log_format,
            date_format
        )
        
        # Mutation logger
        self.loggers['mutation'] = self._create_logger(
            'mutation',
            self.log_dir / 'mutation' / 'mutation.log',
            log_format,
            date_format
        )
        
        # Execution logger
        self.loggers['execution'] = self._create_logger(
            'execution',
            self.log_dir / 'execution' / 'execution.log',
            log_format,
            date_format
        )
        
        # Evaluation logger
        self.loggers['evaluation'] = self._create_logger(
            'evaluation',
            self.log_dir / 'evaluation' / 'evaluation.log',
            log_format,
            date_format
        )
        
        # Metrics logger
        self.loggers['metrics'] = self._create_logger(
            'metrics',
            self.log_dir / 'metrics' / 'metrics.log',
            log_format,
            date_format
        )
        
        # Error logger
        self.loggers['errors'] = self._create_logger(
            'errors',
            self.log_dir / 'errors' / 'errors.log',
            log_format,
            date_format,
            level=logging.ERROR
        )
    
    def _create_logger(self, name: str, log_file: Path, 
                      log_format: str, date_format: str,
                      level: int = logging.INFO) -> logging.Logger:
        """
        Create a logger with file and console handlers.
        
        Args:
            name: Logger name
            log_file: Path to log file
            log_format: Log message format
            date_format: Date format
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """
        Log info message to system logger.
        
        Args:
            message: Message to log
        """
        self.loggers['system'].info(message)
    
    def error(self, message: str):
        """
        Log error message to system logger.
        
        Args:
            message: Message to log
        """
        self.loggers['system'].error(message)
    
    def warning(self, message: str):
        """
        Log warning message to system logger.
        
        Args:
            message: Message to log
        """
        self.loggers['system'].warning(message)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a specific logger.
        
        Args:
            name: Logger name
            
        Returns:
            Requested logger
        """
        return self.loggers.get(name, self.loggers['system'])
    
    def log_execution(self, execution_id: str, data: dict):
        """
        Log individual execution data to separate JSON file.
        
        Args:
            execution_id: Unique execution identifier
            data: Execution data dictionary
        """
        import json
        
        execution_log_dir = self.log_dir / 'execution'
        execution_file = execution_log_dir / f"{execution_id}.json"
        
        with open(execution_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

# Global logger instance
_logger_instance: Optional[ExperimentLogger] = None

def get_logger(name: str = 'system') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ExperimentLogger()
    return _logger_instance.get_logger(name)

def setup_logging(log_dir: str = "logs") -> ExperimentLogger:
    """
    Setup the logging system.
    
    Args:
        log_dir: Base directory for log files
        
    Returns:
        ExperimentLogger instance
    """
    global _logger_instance
    _logger_instance = ExperimentLogger(log_dir)
    return _logger_instance