package resampler_test

import (
	"math"
	"math/rand"
	"testing"

	resampler "github.com/382868331/gsb-rational-resampler-20260919"
)

// reference 用短数组直接上采样卷积实现完整定义，作为对照。
func reference(x []float64, h []float64, L, M int) []float64 {
	N := len(x)
	if N == 0 {
		return nil
	}
	K := len(h)
	u := make([]float64, (N-1)*L+K) // 下标 0..(N-1)L+K-1
	for n, v := range x {
		u[n*L] = v
	}
	last := (N-1)*L + K - 1
	var out []float64
	for q := 0; q*M <= last; q++ {
		t := q * M
		sum := 0.0
		for j := 0; j < K && j <= t; j++ {
			sum += h[j] * u[t-j]
		}
		out = append(out, sum)
	}
	return out
}

func assertClose(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		tol := 1e-10 * (1 + math.Abs(want[i]))
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("sample %d: got %v, want %v (tol %v)", i, got[i], want[i], tol)
		}
	}
}

// runChunked 按给定块大小分块 Feed 并 Finish，拼接全部输出。
func runChunked(t *testing.T, r *resampler.Resampler, x []float64, chunks []int) []float64 {
	t.Helper()
	var out []float64
	pos := 0
	for _, c := range chunks {
		if c > len(x)-pos {
			c = len(x) - pos
		}
		o, err := r.Feed(x[pos : pos+c])
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
		out = append(out, o...)
		pos += c
	}
	if pos < len(x) {
		o, err := r.Feed(x[pos:])
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
		out = append(out, o...)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return append(out, tail...)
}

func mustNew(t *testing.T, L, M int, h []float64) *resampler.Resampler {
	t.Helper()
	r, err := resampler.New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return r
}

// 指定用例：L=4,M=1,K=1,N=1 时只输出一个样本。
func TestSingleSampleDomain(t *testing.T) {
	r := mustNew(t, 4, 1, []float64{2})
	out, err := r.Feed([]float64{0.5})
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	assertClose(t, out, []float64{1.0})
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if len(tail) != 0 {
		t.Fatalf("expected empty tail, got %v", tail)
	}
	if r.TotalOutput() != 1 {
		t.Fatalf("TotalOutput = %d, want 1", r.TotalOutput())
	}
}

// K<L 时输出域由 (N-1)L+K-1 决定，Feed 即输出全部，Finish 为空。
func TestKLessThanL(t *testing.T) {
	h := []float64{1, -0.5, 0.25}
	x := []float64{0.3, -0.7, 0.1, 0.9, -0.2}
	r := mustNew(t, 8, 3, h)
	out, err := r.Feed(x)
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if len(tail) != 0 {
		t.Fatalf("K<L: expected empty tail, got %v", tail)
	}
	assertClose(t, out, reference(x, h, 8, 3))
}

// 空块与空输入：Feed(nil) 不输出；N=0 时 Finish 为空；结束后 Feed 报错。
func TestEmptyBlockAndInput(t *testing.T) {
	r := mustNew(t, 3, 2, []float64{1, 1, 1, 1})
	out, err := r.Feed(nil)
	if err != nil || len(out) != 0 {
		t.Fatalf("empty Feed: out=%v err=%v", out, err)
	}
	tail, err := r.Finish()
	if err != nil || len(tail) != 0 {
		t.Fatalf("Finish with N=0: out=%v err=%v", tail, err)
	}
	if _, err := r.Feed([]float64{0.1}); err == nil {
		t.Fatal("Feed after Finish should fail")
	}
}

// L=M=1 时退化为普通 FIR 卷积。
func TestIdentityRate(t *testing.T) {
	h := []float64{0.5, -1, 2}
	x := []float64{0.1, 0.2, 0.3, 0.4, 0.5, -0.5}
	r := mustNew(t, 1, 1, h)
	got := runChunked(t, r, x, []int{2, 1, 3})
	assertClose(t, got, reference(x, h, 1, 1))
}

// 升采样 L=3,M=2，分块与整段一致。
func TestUpsampleChunked(t *testing.T) {
	rng := rand.New(rand.NewSource(20260919))
	h := make([]float64, 17)
	for i := range h {
		h[i] = rng.Float64()*2 - 1
	}
	x := make([]float64, 40)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	r := mustNew(t, 3, 2, h)
	got := runChunked(t, r, x, []int{1, 7, 13, 5, 14})
	assertClose(t, got, reference(x, h, 3, 2))
}

// 降采样 L=2,M=3，分块与整段一致。
func TestDownsampleChunked(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	h := make([]float64, 23)
	for i := range h {
		h[i] = rng.Float64()*2 - 1
	}
	x := make([]float64, 50)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	r := mustNew(t, 2, 3, h)
	got := runChunked(t, r, x, []int{11, 3, 36})
	assertClose(t, got, reference(x, h, 2, 3))
}

// 冲激输入：输出应为 h 按 L/M 抽取的形式，并与参考一致。
func TestImpulse(t *testing.T) {
	h := []float64{1, 2, 3, 4, 5}
	x := []float64{1, 0, 0, 0}
	r := mustNew(t, 2, 1, h)
	got := runChunked(t, r, x, []int{4})
	assertClose(t, got, reference(x, h, 2, 1))
	// 冲激下 y[q]=z[q]=h[q]（M=1），前 3 个应为 h[0],h[1],h[2]。
	want := []float64{1, 2, 3}
	assertClose(t, got[:3], want)
}

// Finish 尾部：K>L 时 Feed 不能输出全部，Finish 补齐剩余部分。
func TestFinishTail(t *testing.T) {
	h := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1} // K=10 > L=3
	x := []float64{0.5, -0.5, 0.25}
	r := mustNew(t, 3, 2, h)
	feedOut, err := r.Feed(x)
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if len(tail) == 0 {
		t.Fatal("expected non-empty tail for K>L")
	}
	assertClose(t, append(feedOut, tail...), reference(x, h, 3, 2))
	// 重复 Finish 为空成功。
	again, err := r.Finish()
	if err != nil || len(again) != 0 {
		t.Fatalf("repeated Finish: out=%v err=%v", again, err)
	}
}

// 坏块原子拒绝：状态不变，后续输出与未收到坏块一致。
func TestBadBlockAtomicReject(t *testing.T) {
	h := []float64{0.25, 0.5, 0.25, -0.5}
	good1 := []float64{0.1, 0.2, 0.3}
	good2 := []float64{0.4, 0.5}

	r := mustNew(t, 2, 1, h)
	if _, err := r.Feed(good1); err != nil {
		t.Fatalf("Feed good1: %v", err)
	}
	inBefore, outBefore := r.TotalInput(), r.TotalOutput()
	for _, bad := range [][]float64{
		{0.1, 1.5},                     // 超幅
		{math.NaN()},                   // 非有限
		{math.Inf(1)},                  // 非有限
		{0.9, -0.9, math.Inf(-1), 0.9}, // 非法值在中间
	} {
		if _, err := r.Feed(bad); err == nil {
			t.Fatalf("bad block %v accepted", bad)
		}
		if r.TotalInput() != inBefore || r.TotalOutput() != outBefore {
			t.Fatalf("state changed after rejecting %v", bad)
		}
	}
	got := runChunked(t, r, good2, []int{2})
	assertClose(t, got, reference(append(good1, good2...), h, 2, 1)[int(outBefore):])
}

// Reset 恢复初始状态，重跑结果与新实例一致。
func TestReset(t *testing.T) {
	h := []float64{1, -0.5, 0.25, 0.125}
	x := []float64{0.3, -0.6, 0.9, 0.1, -0.1}
	r := mustNew(t, 3, 4, h)
	first := runChunked(t, r, x, []int{2, 3})
	r.Reset()
	if r.TotalInput() != 0 || r.TotalOutput() != 0 || r.Buffered() != 0 || r.Finished() {
		t.Fatal("Reset did not restore initial state")
	}
	second := runChunked(t, r, x, []int{5})
	assertClose(t, first, second)
	assertClose(t, second, reference(x, h, 3, 4))
}

// 固定种子随机参数：单切点与多分块都应与参考一致。
func TestRandomChunkedConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	coprime := func(a, b int) bool {
		for b != 0 {
			a, b = b, a%b
		}
		return a == 1
	}
	for trial := 0; trial < 30; trial++ {
		L := 1 + rng.Intn(8)
		M := 1 + rng.Intn(8)
		if !coprime(L, M) {
			continue
		}
		K := 1 + rng.Intn(40)
		h := make([]float64, K)
		for i := range h {
			h[i] = rng.Float64()*32 - 16
		}
		N := 1 + rng.Intn(30)
		x := make([]float64, N)
		for i := range x {
			x[i] = rng.Float64()*2 - 1
		}
		want := reference(x, h, L, M)

		// 单切点：前 cut 个一块，其余一块。
		cut := rng.Intn(N + 1)
		r := mustNew(t, L, M, h)
		got := runChunked(t, r, x, []int{cut})
		assertClose(t, got, want)

		// 随机小块（含空块）。
		r.Reset()
		var chunks []int
		for sum := 0; sum < N; {
			c := rng.Intn(5)
			chunks = append(chunks, c)
			sum += c
		}
		got = runChunked(t, r, x, chunks)
		assertClose(t, got, want)
	}
}

// 计数与缓冲上界。
func TestCountsAndBufferBound(t *testing.T) {
	h := make([]float64, 257)
	for i := range h {
		h[i] = 0.01
	}
	r := mustNew(t, 1, 1, h)
	x := make([]float64, 1000)
	for i := range x {
		x[i] = 0.5
	}
	if _, err := r.Feed(x[:512]); err != nil {
		t.Fatal(err)
	}
	if r.Buffered() > r.BufferedBound() {
		t.Fatalf("Buffered %d exceeds bound %d", r.Buffered(), r.BufferedBound())
	}
	if _, err := r.Feed(x[512:]); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Finish(); err != nil {
		t.Fatal(err)
	}
	if r.TotalInput() != 1000 {
		t.Fatalf("TotalInput = %d, want 1000", r.TotalInput())
	}
	// L=M=1 时完整输出长度 = N+K-1。
	if r.TotalOutput() != int64(1000+257-1) {
		t.Fatalf("TotalOutput = %d, want %d", r.TotalOutput(), 1000+257-1)
	}
	if r.Buffered() > r.BufferedBound() {
		t.Fatalf("Buffered %d exceeds bound %d", r.Buffered(), r.BufferedBound())
	}
}

// 构造参数校验。
func TestConstructorValidation(t *testing.T) {
	ok := []float64{1, 1}
	if _, err := resampler.New(0, 1, ok); err == nil {
		t.Fatal("L=0 accepted")
	}
	if _, err := resampler.New(1, 65, ok); err == nil {
		t.Fatal("M=65 accepted")
	}
	if _, err := resampler.New(2, 4, ok); err == nil {
		t.Fatal("non-coprime L,M accepted")
	}
	if _, err := resampler.New(1, 1, nil); err == nil {
		t.Fatal("K=0 accepted")
	}
	if _, err := resampler.New(1, 1, make([]float64, 258)); err == nil {
		t.Fatal("K=258 accepted")
	}
	if _, err := resampler.New(1, 1, []float64{17}); err == nil {
		t.Fatal("|h|>16 accepted")
	}
	if _, err := resampler.New(1, 1, []float64{math.NaN()}); err == nil {
		t.Fatal("NaN coefficient accepted")
	}
}
