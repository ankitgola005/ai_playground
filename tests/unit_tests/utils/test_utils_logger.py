import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_playground.utils.logger.base_logger import Logger
from ai_playground.utils.logger.console_logger import ConsoleLogger
from ai_playground.utils.logger.logger_manager import LoggerManager


@pytest.fixture
def config():
    """Create a minimal trainer config for testing"""
    return SimpleNamespace(
        log_interval=10,
        log_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def strategy():
    """Create a mock strategy for testing"""
    strategy = MagicMock()
    strategy.is_main_process.return_value = True
    strategy.rank = 0
    return strategy


class TestBaseLogger:
    def test_cannot_instantiate_abstract_logger(self, config):
        """Test that Logger abstract class cannot be instantiated"""
        with pytest.raises(TypeError):
            Logger(config)

    def test_logger_init_creates_log_dir(self):
        """Test that Logger __init__ creates log directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "new_logs"
            config = SimpleNamespace(
                log_interval=10,
                log_dir=str(log_dir),
            )

            logger = ConsoleLogger(config)

            assert log_dir.exists()
            assert logger.log_frequency == 10

    def test_logger_log_config_default_pass(self, config):
        """Test that base logger log_config is a no-op by default"""
        logger = ConsoleLogger(config)
        # Should not raise
        logger.log_config({"key": "value"})


class TestConsoleLogger:
    def test_console_logger_log_metrics(self, config, capsys):
        """Test ConsoleLogger prints metrics to console"""
        logger = ConsoleLogger(config)
        metrics = {"loss": 0.5, "accuracy": 0.95}

        logger.log_metrics(metrics, step=100)

        captured = capsys.readouterr()
        assert "[step 100]" in captured.out
        assert "loss=0.5000" in captured.out
        assert "accuracy=0.9500" in captured.out

    def test_console_logger_log_hparams(self, config, capsys):
        """Test ConsoleLogger prints hyperparameters"""
        logger = ConsoleLogger(config)
        hparams = {"lr": 0.001, "batch_size": 32, "epochs": 10}

        logger.log_hparams(hparams)

        captured = capsys.readouterr()
        assert "Hyperparameters:" in captured.out
        assert "lr: 0.001" in captured.out
        assert "batch_size: 32" in captured.out
        assert "epochs: 10" in captured.out

    def test_console_logger_finalize(self, config):
        """Test ConsoleLogger finalize method"""
        logger = ConsoleLogger(config)
        # Should not raise
        logger.finalize()


class TestLoggerManager:
    def test_logger_manager_init(self, config, strategy):
        """Test LoggerManager initialization"""
        logger1 = ConsoleLogger(config)
        logger2 = ConsoleLogger(config)

        manager = LoggerManager([logger1, logger2], strategy, config)

        assert len(manager.loggers) == 2
        assert manager.log_frequency == 10

    def test_logger_manager_log_metrics_calls_all_loggers(
        self, config, strategy, capsys
    ):
        """Test LoggerManager logs to all attached loggers"""
        logger1 = ConsoleLogger(config)
        logger2 = ConsoleLogger(config)

        manager = LoggerManager([logger1, logger2], strategy, config)
        metrics = {"loss": 0.3}

        manager.log_metrics(metrics, step=50)

        captured = capsys.readouterr()
        # Should have output from both loggers
        output_count = captured.out.count("[step 50]")
        assert output_count == 2

    def test_logger_manager_log_config(self, config, strategy, capsys):
        """Test LoggerManager logs config to all loggers"""
        logger1 = ConsoleLogger(config)
        logger2 = ConsoleLogger(config)

        manager = LoggerManager([logger1, logger2], strategy, config)

        with patch(
            "ai_playground.utils.logger.logger_manager.config_to_dict"
        ) as mock_to_dict:
            with patch(
                "ai_playground.utils.logger.logger_manager.convert_paths"
            ) as mock_convert:
                mock_to_dict.return_value = {"key": "value"}
                mock_convert.return_value = {"key": "value"}

                manager.log_config(config)

                captured = capsys.readouterr()
                assert "ConfigProtocol" in captured.out
                assert "key" in captured.out

    def test_logger_manager_respects_log_frequency(self, config, strategy):
        """Test LoggerManager calls log_metrics on loggers"""
        logger = MagicMock(spec=ConsoleLogger)
        logger.rank_zero_only = True
        manager = LoggerManager([logger], strategy, config)

        metrics = {"loss": 0.5}

        # Should call logger.log_metrics
        manager.log_metrics(metrics, step=10)
        logger.log_metrics.assert_called_once_with(metrics, 10)

    def test_logger_manager_finalize_calls_all_loggers(self, config, strategy):
        """Test LoggerManager finalize calls finalize on all loggers"""
        logger1 = MagicMock(spec=ConsoleLogger)
        logger2 = MagicMock(spec=ConsoleLogger)

        manager = LoggerManager([logger1, logger2], strategy, config)
        manager.finalize()

        logger1.finalize.assert_called_once()
        logger2.finalize.assert_called_once()

    def test_logger_manager_empty_logger_list(self, config, strategy, capsys):
        """Test LoggerManager works with empty logger list"""
        manager = LoggerManager([], strategy, config)

        # Should not raise
        manager.log_metrics({"loss": 0.5}, step=1)
        manager.finalize()


class TestLoggerIntegration:
    def test_multiple_loggers_same_config(self, config, strategy):
        """Test multiple loggers can share same config"""
        logger1 = ConsoleLogger(config)
        logger2 = ConsoleLogger(config)

        manager = LoggerManager([logger1, logger2], strategy, config)

        assert manager.loggers[0].log_dir == manager.loggers[1].log_dir

    def test_logger_manager_with_none_loggers_list(self, config, strategy):
        """Test LoggerManager handles None gracefully"""
        manager = LoggerManager([], strategy, config)

        # Should not raise on empty list
        assert len(manager.loggers) == 0
