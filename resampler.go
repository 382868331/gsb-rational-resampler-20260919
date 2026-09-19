// Package resampler 实现有理采样率 L/M 的实数单通道 FIR 分块重采样。
//
// 数学定义（x 为输入，h 为长度 K 的 FIR 系数）：
//
//	u[nL] = x[n]，其余位置为 0
//	z[t]  = sum(j=0..K-1, h[j] * u[t-j])
//	y[q]  = z[qM]
//
// 累计收到 N>0 个输入时，完整输出为所有满足 0<=qM<=(N-1)L+K-1 的 y[q]；
// N=0 时输出为空序列，定义域外的输入一律视为 0。
//
// 流式语义：每次 Feed 只输出尚未输出且 qM<=min(NL-1,(N-1)L+K-1) 的样本；
// Finish 按完整定义补齐尾部；重复 Finish 返回空成功；Finish 后 Feed 报错；
// Reset 恢复初始状态。实现使用多相系数与有限历史，不构造 L 倍上采样数组，
// 也不保存全部历史。
package resampler

import (
	"errors"
	"fmt"
	"math"
)

// 参数取值上界。
const (
	MaxFactor = 64  // L、M 的最大值
	MaxTaps   = 257 // FIR 长度 K 的最大值
	MaxCoeff  = 16  // 系数绝对值上界
	MaxSample = 1   // 输入样本绝对值上界
)

var (
	// ErrFinished 表示在 Finish 之后又调用了 Feed。
	ErrFinished = errors.New("resampler: Feed called after Finish")
	// ErrInvalidBlock 表示块内含非有限值或绝对值超过 1 的样本。
	ErrInvalidBlock = errors.New("resampler: block contains non-finite or out-of-range sample")
	// ErrOverflow 表示接受该块会使累计计数溢出 int64。
	ErrOverflow = errors.New("resampler: cumulative input count would overflow int64")
)

// Resampler 是单通道有理采样率重采样器，非并发安全。
type Resampler struct {
	L, M, K int
	h       []float64

	hist     []float64 // hist[i] = x[histBase+i]，仅保留后续输出仍需的历史
	histBase int64
	totalIn  int64 // 累计输入样本数 N
	nextOut  int64 // 下一个待输出的 q
	totalOut int64 // 累计输出样本数
	finished bool
}

// New 构造重采样器。要求 1<=L,M<=MaxFactor 且互质，1<=len(h)<=MaxTaps，
// 系数均为有限值且绝对值不超过 MaxCoeff。
func New(L, M int, h []float64) (*Resampler, error) {
	if L < 1 || L > MaxFactor || M < 1 || M > MaxFactor {
		return nil, fmt.Errorf("resampler: L,M must be in [1,%d], got L=%d M=%d", MaxFactor, L, M)
	}
	if gcd(L, M) != 1 {
		return nil, fmt.Errorf("resampler: L=%d and M=%d are not coprime", L, M)
	}
	if len(h) < 1 || len(h) > MaxTaps {
		return nil, fmt.Errorf("resampler: FIR length must be in [1,%d], got %d", MaxTaps, len(h))
	}
	for i, c := range h {
		if math.IsNaN(c) || math.IsInf(c, 0) || math.Abs(c) > MaxCoeff {
			return nil, fmt.Errorf("resampler: coefficient h[%d]=%v is non-finite or |h|>%d", i, c, MaxCoeff)
		}
	}
	return &Resampler{L: L, M: M, K: len(h), h: append([]float64(nil), h...)}, nil
}

// Feed 送入一个输入块，返回本次新产生的输出样本。
// 块非法（含非有限值、|x|>1 或计数将溢出）时整体拒绝，状态保持不变。
// 空块合法且不产生输出。
func (r *Resampler) Feed(block []float64) ([]float64, error) {
	if r.finished {
		return nil, ErrFinished
	}
	for i, v := range block {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > MaxSample {
			return nil, fmt.Errorf("%w: index %d value %v", ErrInvalidBlock, i, v)
		}
	}
	if int64(len(block)) > r.maxTotalIn()-r.totalIn {
		return nil, ErrOverflow
	}
	r.hist = append(r.hist, block...)
	r.totalIn += int64(len(block))
	var out []float64
	if r.totalIn > 0 {
		L := int64(r.L)
		bound := min64(r.totalIn*L-1, (r.totalIn-1)*L+int64(r.K)-1)
		out = r.emitUpTo(bound)
	}
	r.trim()
	return out, nil
}

// Finish 按完整定义补齐尾部输出。重复调用返回空成功。
func (r *Resampler) Finish() ([]float64, error) {
	if r.finished {
		return nil, nil
	}
	r.finished = true
	if r.totalIn == 0 {
		return nil, nil
	}
	bound := (r.totalIn-1)*int64(r.L) + int64(r.K) - 1
	out := r.emitUpTo(bound)
	r.trim()
	return out, nil
}

// Reset 恢复初始状态，可重新开始一次完整的分块处理。
func (r *Resampler) Reset() {
	r.hist = r.hist[:0]
	r.histBase = 0
	r.totalIn = 0
	r.nextOut = 0
	r.totalOut = 0
	r.finished = false
}

// TotalInput 返回累计接受的输入样本数。
func (r *Resampler) TotalInput() int64 { return r.totalIn }

// TotalOutput 返回累计输出的样本数。
func (r *Resampler) TotalOutput() int64 { return r.totalOut }

// Buffered 返回当前保留的历史输入样本数，不超过 BufferedBound。
func (r *Resampler) Buffered() int { return len(r.hist) }

// BufferedBound 返回历史缓冲的上界，只与 K、L 相关。
func (r *Resampler) BufferedBound() int { return (r.K-1)/r.L + 1 }

// Finished 报告是否已调用 Finish。
func (r *Resampler) Finished() bool { return r.finished }

// maxTotalIn 是保证 NL-1、(N-1)L+K-1 及终止条件的 qM（最大 bound+M）
// 都不溢出 int64 的最大累计输入数，因此预留 L 与 M 的余量。
func (r *Resampler) maxTotalIn() int64 {
	return (math.MaxInt64-int64(r.K)+1-int64(r.M)-int64(r.L))/int64(r.L) + 1
}

// emitUpTo 输出所有尚未输出且 qM<=bound 的样本。
func (r *Resampler) emitUpTo(bound int64) []float64 {
	M := int64(r.M)
	var out []float64
	for r.nextOut*M <= bound {
		out = append(out, r.sample(r.nextOut))
		r.nextOut++
	}
	r.totalOut += int64(len(out))
	return out
}

// sample 用多相分解计算 y[q]：y[q] = sum(n, x[n] * h[qM-nL])，
// 其中只有 j=qM-nL 满足 j%L == qM%L 且 0<=j<K 的系数参与。
func (r *Resampler) sample(q int64) float64 {
	qM := q * int64(r.M)
	L := int64(r.L)
	sum := 0.0
	for j := qM % L; j < int64(r.K); j += L {
		n := (qM - j) / L
		if n < 0 {
			break
		}
		if n >= r.totalIn {
			continue // 定义域外的输入为 0
		}
		sum += r.h[j] * r.hist[n-r.histBase]
	}
	return sum
}

// trim 丢弃后续输出不再需要的历史样本。
// 下一个输出 q=nextOut 需要的最小输入下标为 ceil((qM-K+1)/L)，
// 该下标随 q 单调不减，因此更早的历史可以安全丢弃。
func (r *Resampler) trim() {
	nMin := ceilDiv(r.nextOut*int64(r.M)-int64(r.K)+1, int64(r.L))
	if nMin < 0 {
		nMin = 0
	}
	if nMin > r.totalIn {
		nMin = r.totalIn
	}
	if d := nMin - r.histBase; d > 0 {
		copy(r.hist, r.hist[d:])
		r.hist = r.hist[:len(r.hist)-int(d)]
		r.histBase = nMin
	}
}

// ceilDiv 计算 ceil(a/b)，要求 b>0。a>=-257 且 a<=MaxInt64-1，不会溢出。
func ceilDiv(a, b int64) int64 {
	q := a / b
	if a%b != 0 && a >= 0 {
		q++
	}
	return q
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
