package resampler

import (
	"errors"
	"math"
	"testing"
)

// 溢出保护：把累计计数直接放到上界附近，验证 Feed 在溢出前整体拒绝。
func TestOverflowRejected(t *testing.T) {
	r, err := New(64, 1, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	limit := r.maxTotalIn()
	if limit <= 0 || limit > math.MaxInt64/2 {
		t.Fatalf("unexpected limit %d", limit)
	}
	r.totalIn = limit // 白盒：直接置于上界
	if _, err := r.Feed([]float64{0.1}); !errors.Is(err, ErrOverflow) {
		t.Fatalf("expected ErrOverflow, got %v", err)
	}
	if r.totalIn != limit {
		t.Fatal("state changed after overflow rejection")
	}
	// 上界处仍可接受空块之外的拒绝判定：差值为 0 时任何非空块都被拒。
	if _, err := r.Feed([]float64{0.1, 0.2}); !errors.Is(err, ErrOverflow) {
		t.Fatalf("expected ErrOverflow, got %v", err)
	}
}
