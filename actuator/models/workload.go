package models

type Workload struct {
	ID    string
	Label int // 0 = private, 1 = public
	Kind  string
}
