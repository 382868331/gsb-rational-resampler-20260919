// demo 演示分块重采样与整段处理一致、Finish 尾部补齐，以及一次实际触发的失败。
package main

import (
	"fmt"
	"math"
	"time"

	resampler "github.com/382868331/gsb-rational-resampler-20260919"
)

func main() {
	start := time.Now()

	// 信号：两个正弦叠加，|x|<=1；L=3/M=2 升采样，K=13 的 FIR。
	const (
		L, M = 3, 2
		N    = 60
	)
	h := []float64{0.05, -0.1, 0.2, -0.4, 0.8, 1.5, 2.0, 1.5, 0.8, -0.4, 0.2, -0.1, 0.05}
	x := make([]float64, N)
	for n := range x {
		x[n] = 0.6*math.Sin(2*math.Pi*float64(n)/25) + 0.3*math.Sin(2*math.Pi*float64(n)/7)
	}

	// 整段一次处理。
	oneShot, _ := resampler.New(L, M, h)
	var whole []float64
	if out, err := oneShot.Feed(x); err == nil {
		whole = out
	}
	tailWhole, _ := oneShot.Finish()
	feedCount := len(whole)
	whole = append(whole, tailWhole...)

	// 分块处理：块长 1、7、16、其余。
	chunked, _ := resampler.New(L, M, h)
	var parts []float64
	pos := 0
	for _, c := range []int{1, 7, 16, N - 24} {
		out, err := chunked.Feed(x[pos : pos+c])
		if err != nil {
			fmt.Println("Feed 失败:", err)
			return
		}
		parts = append(parts, out...)
		pos += c
	}
	bufferedBeforeFinish := chunked.Buffered()
	tail, _ := chunked.Finish()
	parts = append(parts, tail...)

	// 比较。
	maxDiff := 0.0
	for i := range whole {
		if d := math.Abs(whole[i] - parts[i]); d > maxDiff {
			maxDiff = d
		}
	}
	fmt.Printf("[正常] L=%d M=%d K=%d，输入 %d 个样本\n", L, M, len(h), N)
	fmt.Printf("  整段输出 %d 个样本（Feed %d + Finish 尾部 %d）\n", len(whole), feedCount, len(tailWhole))
	fmt.Printf("  分块输出 %d 个样本（Finish 尾部 %d），与整段最大偏差 %.3e\n", len(parts), len(tail), maxDiff)
	fmt.Printf("  计数：输入 %d，输出 %d，Finish 前历史缓冲 %d（上界 %d）\n",
		chunked.TotalInput(), chunked.TotalOutput(), bufferedBeforeFinish, chunked.BufferedBound())
	fmt.Printf("  前 6 个输出样本：")
	for i := 0; i < 6 && i < len(parts); i++ {
		fmt.Printf("%.6f ", parts[i])
	}
	fmt.Println()
	if len(whole) != len(parts) || maxDiff > 1e-10 {
		fmt.Println("  结论：分块与整段不一致！")
	} else {
		fmt.Println("  结论：分块与整段一致，尾部已补齐")
	}

	// 实际触发一个失败：块内含 |x|>1 的样本，整体拒绝且状态不变。
	bad, _ := resampler.New(L, M, h)
	if _, err := bad.Feed(x[:10]); err != nil {
		fmt.Println("意外失败:", err)
		return
	}
	inBefore := bad.TotalInput()
	_, err := bad.Feed([]float64{0.2, 1.7, -0.2})
	fmt.Printf("\n[失败] 送入含 1.7（|x|>1）的块：%v\n", err)
	fmt.Printf("  状态未变：累计输入 %d -> %d\n", inBefore, bad.TotalInput())
	// 拒绝后仍可继续正常处理并结束。
	if _, err := bad.Feed(x[10:]); err != nil {
		fmt.Println("意外失败:", err)
		return
	}
	if _, err := bad.Finish(); err != nil {
		fmt.Println("意外失败:", err)
		return
	}
	if _, err := bad.Feed(x[:1]); err != nil {
		fmt.Printf("  Finish 后再 Feed：%v\n", err)
	}
	fmt.Printf("  坏块拒绝后完整输出 %d 个样本，与整段一致：%v\n",
		bad.TotalOutput(), bad.TotalOutput() == int64(len(whole)))

	fmt.Printf("\n演示耗时 %.2f 秒\n", time.Since(start).Seconds())
}
