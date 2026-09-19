package resampler

import (
	"math"
	"math/rand/v2"
	"testing"
)

// reference 用短数组直接上采样卷积计算完整定义:
// u[nL]=x[n];z[t]=sum_j h[j]u[t-j];y[q]=z[qM],0<=qM<=(N-1)L+K-1。
func reference(L, M int, h, x []float64) []float64 {
	N := len(x)
	if N == 0 {
		return nil
	}
	K := len(h)
	u := make([]float64, (N-1)*L+1)
	for n, v := range x {
		u[n*L] = v
	}
	z := make([]float64, (N-1)*L+K)
	for t := range z {
		for j := 0; j < K && j <= t; j++ {
			if t-j < len(u) { // 域外输入为 0
				z[t] += h[j] * u[t-j]
			}
		}
	}
	var y []float64
	for q := 0; q*M <= (N-1)*L+K-1; q++ {
		y = append(y, z[q*M])
	}
	return y
}

func tol(ref float64) float64 { return 1e-10 * (1 + math.Abs(ref)) }

func checkClose(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol(want[i]) {
			t.Fatalf("y[%d]: got %v, want %v (tol %v)", i, got[i], want[i], tol(want[i]))
		}
	}
}

// runWhole 一次性 Feed 全部输入再 Finish。
func runWhole(t *testing.T, L, M int, h, x []float64) []float64 {
	t.Helper()
	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	out, err := r.Feed(x)
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return append(out, tail...)
}

// runChunked 按 chunkSizes 分块 Feed(允许 0 长度块),最后 Finish。
func runChunked(t *testing.T, L, M int, h, x []float64, chunkSizes []int) []float64 {
	t.Helper()
	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var out []float64
	pos := 0
	for _, c := range chunkSizes {
		if pos >= len(x) {
			break
		}
		if pos+c > len(x) {
			c = len(x) - pos
		}
		blk, err := r.Feed(x[pos : pos+c])
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
		out = append(out, blk...)
		pos += c
	}
	if pos < len(x) {
		blk, err := r.Feed(x[pos:])
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
		out = append(out, blk...)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return append(out, tail...)
}

// TestSpecCaseL4M1K1N1 覆盖规格用例:L=4,M=1,K=1,N=1 时只输出一个样本。
func TestSpecCaseL4M1K1N1(t *testing.T) {
	r, err := New(4, 1, []float64{2.5})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	out, err := r.Feed([]float64{0.5})
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected exactly 1 output sample, got %d (%v)", len(out), out)
	}
	if want := 2.5 * 0.5; out[0] != want {
		t.Fatalf("got %v, want %v", out[0], want)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if len(tail) != 0 {
		t.Fatalf("expected empty tail, got %v", tail)
	}
}

// TestKLessThanLDomain 覆盖 K<L 输出域:Feed 的边界 min(NL-1,(N-1)L+K-1)
// 此时等于完整定义边界,Feed 即输出全部样本,Finish 为空。
func TestKLessThanLDomain(t *testing.T) {
	L, M := 4, 1
	h := []float64{1, -2}
	x := []float64{0.25, -0.5, 1}
	want := reference(L, M, h, x)
	// 完整定义边界 (N-1)L+K-1 = 9,共 10 个输出。
	if len(want) != 10 {
		t.Fatalf("reference length: got %d, want 10", len(want))
	}
	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
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
	checkClose(t, out, want)
}

// TestEmptyBlockAndEmptyInput 覆盖空块与空输入。
func TestEmptyBlockAndEmptyInput(t *testing.T) {
	r, err := New(3, 2, []float64{1, 2, 3})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, blk := range [][]float64{nil, {}, {}} {
		out, err := r.Feed(blk)
		if err != nil {
			t.Fatalf("Feed(empty): %v", err)
		}
		if len(out) != 0 {
			t.Fatalf("empty block produced %v", out)
		}
	}
	if s := r.Stats(); s.InputSamples != 0 || s.OutputSamples != 0 || s.BufferedSamples != 0 {
		t.Fatalf("stats after empty feeds: %+v", s)
	}
	// 空输入整体:Finish 输出空。
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if len(tail) != 0 {
		t.Fatalf("Finish with N=0 produced %v", tail)
	}

	// 空块穿插在真实数据中,结果应与整段一致。
	x := []float64{0.1, -0.2, 0.3, -0.4, 0.5}
	got := runChunked(t, 3, 2, []float64{1, 2, 3}, x, []int{0, 2, 0, 0, 3, 0})
	checkClose(t, got, reference(3, 2, []float64{1, 2, 3}, x))
}

// TestIdentityLM 覆盖 L=M=1(纯 FIR 卷积)。
func TestIdentityLM(t *testing.T) {
	h := []float64{0.5, -1, 2, 0.25, 1}
	x := []float64{0.1, 0.2, -0.3, 0.4, 0.9, -0.8, 0.0, 1.0}
	checkClose(t, runWhole(t, 1, 1, h, x), reference(1, 1, h, x))
	checkClose(t, runChunked(t, 1, 1, h, x, []int{3, 1, 4}), reference(1, 1, h, x))
}

// TestUpAndDownSampling 覆盖升采样、降采样与混合比率。
func TestUpAndDownSampling(t *testing.T) {
	x := []float64{0.3, -0.7, 0.1, 0.9, -0.2, 0.5, 0.0, -1.0, 0.8, 0.4}
	cases := []struct {
		L, M int
		h    []float64
	}{
		{4, 1, []float64{1, 0.5, -0.25, 2}},                     // 升采样
		{1, 4, []float64{1, -1, 0.5}},                           // 降采样
		{3, 2, []float64{0.75, -0.5, 1.25, 0.5, -1}},            // 3/2
		{2, 3, []float64{1, 1, 1, 1, 1, 1, 1}},                  // 2/3
		{5, 3, []float64{-2, 1, 0.5, 0.25, -0.125, 3, -1, 0.5}}, // 5/3
	}
	for _, c := range cases {
		want := reference(c.L, c.M, c.h, x)
		checkClose(t, runWhole(t, c.L, c.M, c.h, x), want)
		checkClose(t, runChunked(t, c.L, c.M, c.h, x, []int{1, 2, 3, 4}), want)
	}
}

// TestImpulse 覆盖冲激输入:输出应与 h 的相应抽样一致。
func TestImpulse(t *testing.T) {
	L, M := 2, 3
	h := []float64{1, -2, 3, -4, 5}
	x := []float64{1, 0, 0, 0, 0}
	got := runWhole(t, L, M, h, x)
	checkClose(t, got, reference(L, M, h, x))
	// 冲激在 n=0:u 仅在 t=0 非零,故 y[q]=h[qM](qM<K),否则 0。
	for q, v := range got {
		qM := q * M
		var want float64
		if qM < len(h) {
			want = h[qM]
		}
		if v != want {
			t.Fatalf("y[%d]: got %v, want %v", q, v, want)
		}
	}
}

// TestFinishTail 覆盖 Finish 尾部:K>L 时 Feed 边界 NL-1 小于完整边界,
// 尾部由 Finish 补齐;重复 Finish 为空成功;结束后 Feed 报错。
func TestFinishTail(t *testing.T) {
	L, M := 2, 1
	h := []float64{1, 2, 3, 4, 5, 6}
	x := []float64{0.5, -0.5, 0.25}
	want := reference(L, M, h, x)

	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	out, err := r.Feed(x)
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	// Feed 边界 NL-1=5 → 6 个样本;完整边界 (N-1)L+K-1=9 → 10 个样本。
	if len(out) != 6 {
		t.Fatalf("Feed produced %d samples, want 6", len(out))
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if len(tail) != 4 {
		t.Fatalf("tail length %d, want 4", len(tail))
	}
	checkClose(t, append(out, tail...), want)

	// 重复 Finish:空成功。
	again, err := r.Finish()
	if err != nil {
		t.Fatalf("second Finish: %v", err)
	}
	if len(again) != 0 {
		t.Fatalf("second Finish produced %v", again)
	}
	// 结束后 Feed 报错。
	if _, err := r.Feed([]float64{0.1}); err != ErrFinished {
		t.Fatalf("Feed after Finish: got %v, want ErrFinished", err)
	}
}

// TestBadBlockAtomicReject 覆盖坏块整体拒绝且状态不变。
func TestBadBlockAtomicReject(t *testing.T) {
	L, M := 3, 2
	h := []float64{1, -0.5, 2}
	good1 := []float64{0.1, 0.2, 0.3}
	good2 := []float64{0.4, -0.5}

	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	out1, err := r.Feed(good1)
	if err != nil {
		t.Fatalf("Feed good1: %v", err)
	}
	before := r.Stats()

	badBlocks := [][]float64{
		{0.1, math.NaN(), 0.2},
		{math.Inf(1)},
		{math.Inf(-1)},
		{0.1, 1.0000001},
		{-1.5},
	}
	for _, blk := range badBlocks {
		if _, err := r.Feed(blk); err == nil {
			t.Fatalf("bad block %v accepted", blk)
		}
		if got := r.Stats(); got != before {
			t.Fatalf("state changed after bad block %v: before %+v after %+v", blk, before, got)
		}
	}

	// 状态未变,后续合法块继续处理,整体结果等于只喂合法数据的参考。
	out2, err := r.Feed(good2)
	if err != nil {
		t.Fatalf("Feed good2: %v", err)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	got := append(append(out1, out2...), tail...)
	checkClose(t, got, reference(L, M, h, append(good1, good2...)))
}

// TestNewInvalidParams 覆盖构造参数校验。
func TestNewInvalidParams(t *testing.T) {
	good := []float64{1, 2, 3}
	cases := []struct {
		L, M int
		h    []float64
	}{
		{0, 1, good}, {1, 0, good}, {65, 1, good}, {1, 65, good}, {-1, 1, good},
		{2, 4, good}, {6, 3, good}, // 非互质
		{1, 1, nil}, {1, 1, make([]float64, 258)},
		{1, 1, []float64{1, math.NaN()}},
		{1, 1, []float64{math.Inf(1)}},
		{1, 1, []float64{16.000001}},
		{1, 1, []float64{-17}},
	}
	for i, c := range cases {
		if _, err := New(c.L, c.M, c.h); err == nil {
			t.Fatalf("case %d: New(%d,%d,h) unexpectedly succeeded", i, c.L, c.M)
		}
	}
	// 边界合法值应成功。
	if _, err := New(64, 63, make([]float64, 257)); err != nil {
		t.Fatalf("boundary params rejected: %v", err)
	}
	if _, err := New(1, 1, []float64{16}); err != nil {
		t.Fatalf("|h|=16 rejected: %v", err)
	}
}

// TestReset 覆盖 Reset 恢复初始状态。
func TestReset(t *testing.T) {
	L, M := 3, 2
	h := []float64{1, -2, 0.5, 1}
	x := []float64{0.1, 0.2, 0.3, 0.4, 0.5}

	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := r.Feed(x[:3]); err != nil {
		t.Fatalf("Feed: %v", err)
	}
	r.Reset()
	if s := r.Stats(); s.InputSamples != 0 || s.OutputSamples != 0 || s.BufferedSamples != 0 {
		t.Fatalf("stats after Reset: %+v", s)
	}
	// Reset 后重跑应与全新实例一致(包括 Finish 后 Reset 的场景)。
	var got []float64
	for _, blk := range [][]float64{x[:2], x[2:]} {
		o, err := r.Feed(blk)
		if err != nil {
			t.Fatalf("Feed: %v", err)
		}
		got = append(got, o...)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	got = append(got, tail...)
	checkClose(t, got, reference(L, M, h, x))

	r.Reset()
	if _, err := r.Feed(x); err != nil {
		t.Fatalf("Feed after Finish+Reset: %v", err)
	}
}

// TestChunkedEqualsReference 固定种子随机验证:随机互质 (L,M)、随机 K、
// 随机分块(含空块与单切点),分块流式结果与直接上采样卷积参考一致。
func TestChunkedEqualsReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x5eed, 0x2026))
	for trial := 0; trial < 60; trial++ {
		var L, M int
		for {
			L = 1 + rng.IntN(64)
			M = 1 + rng.IntN(64)
			if gcd(L, M) == 1 {
				break
			}
		}
		K := 1 + rng.IntN(257)
		h := make([]float64, K)
		for j := range h {
			h[j] = 32*rng.Float64() - 16
		}
		N := 1 + rng.IntN(60)
		x := make([]float64, N)
		for n := range x {
			x[n] = 2*rng.Float64() - 1
		}
		var chunks []int
		if trial%2 == 0 {
			chunks = []int{N / 2} // 单切点
		} else {
			for rest := N; rest > 0; {
				c := rng.IntN(9) // 允许 0 长度块
				chunks = append(chunks, c)
				rest -= c
			}
		}
		want := reference(L, M, h, x)
		got := runChunked(t, L, M, h, x, chunks)
		if len(got) != len(want) {
			t.Fatalf("trial %d (L=%d M=%d K=%d N=%d): length %d, want %d", trial, L, M, K, N, len(got), len(want))
		}
		checkClose(t, got, want)
	}
}

// TestStats 覆盖累计计数与缓冲上界(与 K、L 相关)。
func TestStats(t *testing.T) {
	L, M, K := 4, 1, 9
	h := make([]float64, K)
	for j := range h {
		h[j] = 1
	}
	r, err := New(L, M, h)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	x := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	out, err := r.Feed(x)
	if err != nil {
		t.Fatalf("Feed: %v", err)
	}
	s := r.Stats()
	if s.InputSamples != 5 {
		t.Fatalf("InputSamples=%d, want 5", s.InputSamples)
	}
	if s.OutputSamples != int64(len(out)) {
		t.Fatalf("OutputSamples=%d, want %d", s.OutputSamples, len(out))
	}
	maxBuf := int64((K-1)/L + 2)
	if s.BufferedSamples != 4 || s.BufferedSamples > maxBuf {
		t.Fatalf("BufferedSamples=%d, want min(5,%d)=4", s.BufferedSamples, maxBuf)
	}
	tail, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	s = r.Stats()
	if s.OutputSamples != int64(len(out)+len(tail)) {
		t.Fatalf("OutputSamples after Finish=%d, want %d", s.OutputSamples, len(out)+len(tail))
	}
	if s.BufferedSamples > maxBuf {
		t.Fatalf("BufferedSamples=%d exceeds bound %d", s.BufferedSamples, maxBuf)
	}
	// 长输入后缓冲仍受 (K-1)/L+2 限制。
	r.Reset()
	big := make([]float64, 1000)
	if _, err := r.Feed(big); err != nil {
		t.Fatalf("Feed: %v", err)
	}
	if got := r.Stats().BufferedSamples; got != maxBuf {
		t.Fatalf("BufferedSamples=%d, want %d", got, maxBuf)
	}
}

// TestOverflowGuard 覆盖计数溢出前拒绝(白盒:直接抬高累计计数)。
func TestOverflowGuard(t *testing.T) {
	r, err := New(64, 1, []float64{1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r.nIn = math.MaxInt64 - 10 // 接近溢出
	if _, err := r.Feed([]float64{0.1, 0.2}); err != ErrOverflow {
		t.Fatalf("got %v, want ErrOverflow", err)
	}
	if r.nIn != math.MaxInt64-10 {
		t.Fatalf("state changed after overflow rejection")
	}
}

// TestFeedAfterFinishAndRepeatFinish 覆盖结束后的行为。
func TestFeedAfterFinishAndRepeatFinish(t *testing.T) {
	r, err := New(1, 1, []float64{1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := r.Feed([]float64{0.5}); err != nil {
		t.Fatalf("Feed: %v", err)
	}
	if _, err := r.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if _, err := r.Feed([]float64{0.5}); err != ErrFinished {
		t.Fatalf("Feed after Finish: got %v, want ErrFinished", err)
	}
	for i := 0; i < 3; i++ {
		out, err := r.Finish()
		if err != nil {
			t.Fatalf("repeat Finish: %v", err)
		}
		if len(out) != 0 {
			t.Fatalf("repeat Finish produced %v", out)
		}
	}
}
