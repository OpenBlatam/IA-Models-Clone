package utils

import (
	"log"
	"os"
	"path/filepath"
	"time"
)

// LogLevel represents logging levels
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarning
	LogLevelError
	LogLevelCritical
)

// Logger provides structured logging
type Logger struct {
	level  LogLevel
	logger *log.Logger
	file   *os.File
}

// NewLogger creates a new logger instance
func NewLogger(level LogLevel, logFile string) (*Logger, error) {
	var file *os.File
	var err error

	if logFile != "" {
		// Create log directory if it doesn't exist
		dir := filepath.Dir(logFile)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}

		file, err = os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			return nil, err
		}
	}

	logger := log.New(file, "", log.LstdFlags|log.Lmicroseconds)
	if file == nil {
		logger.SetOutput(os.Stdout)
	}

	return &Logger{
		level:  level,
		logger: logger,
		file:   file,
	}, nil
}

// Close closes the logger and file
func (l *Logger) Close() error {
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}

// Debug logs a debug message
func (l *Logger) Debug(format string, v ...interface{}) {
	if l.level <= LogLevelDebug {
		l.logger.Printf("[DEBUG] "+format, v...)
	}
}

// Info logs an info message
func (l *Logger) Info(format string, v ...interface{}) {
	if l.level <= LogLevelInfo {
		l.logger.Printf("[INFO] "+format, v...)
	}
}

// Warning logs a warning message
func (l *Logger) Warning(format string, v ...interface{}) {
	if l.level <= LogLevelWarning {
		l.logger.Printf("[WARNING] "+format, v...)
	}
}

// Error logs an error message
func (l *Logger) Error(format string, v ...interface{}) {
	if l.level <= LogLevelError {
		l.logger.Printf("[ERROR] "+format, v...)
	}
}

// Critical logs a critical message
func (l *Logger) Critical(format string, v ...interface{}) {
	if l.level <= LogLevelCritical {
		l.logger.Printf("[CRITICAL] "+format, v...)
	}
}

// LogPerformance logs performance metrics
func (l *Logger) LogPerformance(operation string, duration time.Duration) {
	l.Info("%s took %v", operation, duration)
}

// LogErrorWithContext logs an error with additional context
func (l *Logger) LogErrorWithContext(err error, context map[string]interface{}) {
	contextStr := ""
	for k, v := range context {
		contextStr += k + "=" + toString(v) + " "
	}
	l.Error("Error: %v | Context: %s", err, contextStr)
}

// toString converts a value to string
func toString(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case int:
		return string(rune(val))
	case time.Duration:
		return val.String()
	default:
		return "unknown"
	}
}












