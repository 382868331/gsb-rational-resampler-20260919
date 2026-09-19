// Package resampler 提供实数单通道、有理采样率 L/M 的流式 FIR 重采样。
//
// 数学定义:u[nL]=x[n],其余位置为 0;z[t]=sum(j=0..K-1, h[j]*u[t-j]);
// y[q]=z[qM]。累计收到 N>0 个输入样本时,完整输出为满足
// 0<=qM<=(N-1)L+K-1 的 y[q];N=0 时输出空序列,域外输入视为 0。
//
// 流式语义:累计收到 N>0 个样本时,Feed 只输出尚未输出且
// qM<=min(NL-1,(N-1)L+K-1) 的样本;N=0 不输出。Finish 按完整定义补齐
// 尾部;重复 Finish 返回空成功;Finish 之后 Feed 报错;Reset 恢复初始状态。
//
// 实现使用多相系数与有限历史(环形缓冲,大小 (K-1)/L+2,仅与 K、L 相关),
// 不构造 L 倍上采样数组,也不保存全部输入历史。
package resampler

import (
	"errors"
	"fmt"
	"math"
)

// 参数与输入的取值约束。
const (
	MaxFactor  = 64   // L、M 的上界(含)
	MaxTaps    = 257  // FIR 长度 K 的上界(含)
	MaxAbsIn   = 1.0  // 输入样本绝对值上界(含)
	MaxAbsCoef = 16.0 // 系数绝对值上界(含)
)

// ErrFinished 表示在 Finish 之后又调用了 Feed。
var ErrFinished = errors.New("resampler: Feed called after Finish")

// ErrOverflow 表示继续接收输入会使 int64 计数濒临溢出,该块被整体拒绝。
var ErrOverflow = errors.New("resampler: input block would risk int64 counter overflow")

// Stats 记录累计输入、累计输出与当前缓冲的样本数。
type Stats struct {
	InputSamples    int64 // 累计接收的输入样本数
	OutputSamples   int64 // 累计输出的样本数
	BufferedSamples int64 // 当前历史缓冲中的样本数,上界为 (K-1)/L+2
}

// Resampler 是单通道实数流式重采样器。不是并发安全的。
type Resampler struct {
	L, M, K int
	h       []float64
	taps    [][]int // taps[p]:满足 j≡p (mod L) 的系数下标,升序
	hist    []float64
	nIn     int64 // 累计输入样本数
	nextQ   int64 // 下一个待输出的 q
	nOut    int64 // 累计输出样本数
	done    bool  // 是否已 Finish
}

// New 构造重采样器。要求 1<=L,M<=MaxFactor 且互质,1<=len(h)<=MaxTaps,
// 系数均为有限值且绝对值不超过 MaxAbsCoef。参数非法时返回错误。
func New(L, M int, h []float64) (*Resampler, error) {
	if L < 1 || L > MaxFactor || M < 1 || M > MaxFactor {
		return nil, fmt.Errorf("resampler: L,M must be in [1,%d], got L=%d M=%d", MaxFactor, L, M)
	}
	if gcd(L, M) != 1 {
		return nil, fmt.Errorf("resampler: L=%d and M=%d must be coprime", L, M)
	}
	K := len(h)
	if K < 1 || K > MaxTaps {
		return nil, fmt.Errorf("resampler: len(h) must be in [1,%d], got %d", MaxTaps, K)
	}
	hc := make([]float64, K)
	for i, c := range h {
		if math.IsNaN(c) || math.IsInf(c, 0) || math.Abs(c) > MaxAbsCoef {
			return nil, fmt.Errorf("resampler: invalid coefficient h[%d]=%v (need finite, |h|<=%v)", i, c, MaxAbsCoef)
		}
		hc[i] = c
	}

	taps := make([][]int, L)
	for p := 0; p < L; p++ {
		for j := p; j < K; j += L {
			taps[p] = append(taps[p], j)
		}
	}

	return &Resampler{
		L: L, M: M, K: K,
		h:    hc,
		taps: taps,
		hist: make([]float64, (K-1)/L+2),
	}, nil
}

// Feed 接收一个输入块,返回本次新产生的输出样本(可能为空)。
// 块中样本须为有限值且绝对值不超过 MaxAbsIn;若块非法或会使计数溢出,
// 整个块被拒绝且重采样器状态不变。Finish 之后调用返回 ErrFinished。
func (r *Resampler) Feed(x []float64) ([]float64, error) {
	if r.done {
		return nil, ErrFinished
	}
	for i, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > MaxAbsIn {
			return nil, fmt.Errorf("resampler: invalid input sample x[%d]=%v (need finite, |x|<=%v); block rejected", i, v, MaxAbsIn)
		}
	}
	// 溢出防护:需要 (N-1)*L+K-1+M <= math.MaxInt64 始终成立。
	if len(x) > 0 {
		limit := (math.MaxInt64-int64(r.K)-int64(r.M))/int64(r.L) + 1
		if int64(len(x)) > limit-r.nIn {
			return nil, ErrOverflow
		}
	}

	var out []float64
	for _, v := range x {
		r.hist[int(r.nIn%int64(len(r.hist)))] = v
		r.nIn++
		out = r.emit(out, r.feedBound())
	}
	return out, nil
}

// Finish 按完整定义补齐尾部样本(qM<=(N-1)L+K-1 中尚未输出的部分)。
// N=0 时返回空。重复调用返回空成功。之后 Feed 将返回 ErrFinished。
func (r *Resampler) Finish() ([]float64, error) {
	if r.done {
		return nil, nil
	}
	r.done = true
	if r.nIn == 0 {
		return nil, nil
	}
	bound := (r.nIn-1)*int64(r.L) + int64(r.K) - 1
	return r.emit(nil, bound), nil
}

// Reset 恢复到初始状态(参数与系数保留),可重新开始流式处理。
func (r *Resampler) Reset() {
	r.nIn, r.nextQ, r.nOut = 0, 0, 0
	r.done = false
	for i := range r.hist {
		r.hist[i] = 0
	}
}

// Stats 返回累计输入/输出与当前缓冲样本数。
func (r *Resampler) Stats() Stats {
	buffered := r.nIn
	if int64(len(r.hist)) < buffered {
		buffered = int64(len(r.hist))
	}
	return Stats{InputSamples: r.nIn, OutputSamples: r.nOut, BufferedSamples: buffered}
}

// feedBound 返回 Feed 阶段允许输出的最大 t=qM(含);N=0 时返回 -1。
// 即 min(NL-1, (N-1)L+K-1) = NL-1-max(0,L-K)。
func (r *Resampler) feedBound() int64 {
	if r.nIn == 0 {
		return -1
	}
	bound := r.nIn*int64(r.L) - 1
	if d := int64(r.L - r.K); d > 0 {
		bound -= d
	}
	return bound
}

// emit 输出所有满足 nextQ*M<=bound 的样本,追加到 out 后返回。
func (r *Resampler) emit(out []float64, bound int64) []float64 {
	m := int64(r.M)
	for r.nextQ*m <= bound {
		out = append(out, r.compute(r.nextQ))
		r.nextQ++
		r.nOut++
	}
	return out
}

// compute 用多相系数计算 y[q]:j 取遍满足 j≡qM (mod L) 的下标,
// 对应输入为 x[(qM-j)/L],域外输入为 0。
func (r *Resampler) compute(q int64) float64 {
	t := q * int64(r.M)
	p := t % int64(r.L)
	nMax := t / int64(r.L)
	taps := r.taps[p]
	var y float64
	for i, j := range taps {
		n := nMax - int64(i)
		if n < 0 {
			break
		}
		y += r.h[j] * r.sampleAt(n)
	}
	return y
}

// sampleAt 返回 x[n](n>=0);n 超出已接收范围时按定义取 0。
// 由输出边界可证明 n>=nIn-len(hist),故环形缓冲内的样本必然有效。
func (r *Resampler) sampleAt(n int64) float64 {
	if n >= r.nIn {
		return 0
	}
	return r.hist[int(n%int64(len(r.hist)))]
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
