
package terminal
type any interface{}
const (
	_ = iota
	LogColorRed = "\033[0;31m"
	LogColorBlue = "\033[94m"
	LogColorGreen = "\033[92m"
	LogColorWarning = "\033[93m"
	LogColorFail = "\033[91m"
	LogColorHeader = "\033[95m"
)

const (
	LevelCritical = iota
	LevelWarning
	LevelLog
)