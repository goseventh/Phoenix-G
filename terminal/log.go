package terminal

import (
	"log"
)

func NewLog(tag, text string, levelLog uint8, args ...any) {
	switch levelLog {
	case levelLog:
		log.Printf(
			"["+LogColorHeader+"]"+tag+LogColorGreen+" "+text, args)
	case LevelWarning:
	case LevelCritical:
	}
}

func CloneLog(text string, args ...any) {
	log.Printf(text, args)
}
