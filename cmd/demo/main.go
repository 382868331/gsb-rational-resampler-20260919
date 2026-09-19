// 演示:分块重采样与整段处理一致、Finish 补齐尾部,以及一个实际触发的失败
// (非法块被整体拒绝且状态不变,结束后 Feed 报错)。
package main

import (
	"fmt"
	"math"

	"github.com/382868331/gsb-rational-resampler-20260919/resampler"
)

func main() {
	// ---------- 正常结果:分块 == 整段,Finish 补齐尾部 ----------
	const L, M = 3, 2
	h := []float64{0.1, 0.2, 0.4, 0.8, 1.0, 0.8, 0.4, 0.2, 0.1}

	const N = 96
	x := make([]float64, N)
	for n := range x {
		x[n] = 0.7*math.Sin(2*math.Pi*float64(n)/16) + 0.2*math.Sin(2*math.Pi*float64(n)/5)
	}

	// 整段处理。
	whole, err := resampler.New(L, M, h)
	must(err)
	wholeOut, err := whole.Feed(x)
	must(err)
	wholeTail, err := whole.Finish()
	must(err)
	wholeAll := append(wholeOut, wholeTail...)

	// 分块处理(含空块)。
	chunked, err := resampler.New(L, M, h)
	must(err)
	var chunkedAll []float64
	pos := 0
	for _, c := range []int{1, 13, 0, 40, 7, 35} {
		blk, err := chunked.Feed(x[pos : pos+c])
		must(err)
		chunkedAll = append(chunkedAll, blk...)
		pos += c
	}
	chunkedTail, err := chunked.Finish()
	must(err)
	chunkedAll = append(chunkedAll, chunkedTail...)

	maxDiff := 0.0
	for i := range wholeAll {
		if d := math.Abs(wholeAll[i] - chunkedAll[i]); d > maxDiff {
			maxDiff = d
		}
	}
	fmt.Println("=== 正常结果 ===")
	fmt.Printf("L=%d M=%d K=%d,输入 %d 样本,分块大小 [1 13 0 40 7 35]\n", L, M, len(h), N)
	fmt.Printf("整段输出 %d 样本(Feed %d + Finish 尾部 %d)\n",
		len(wholeAll), len(wholeOut), len(wholeTail))
	fmt.Printf("分块输出 %d 样本,与整段最大偏差 %.3e\n", len(chunkedAll), maxDiff)
	fmt.Printf("前 6 个输出样本: %.6f\n", wholeAll[:6])
	fmt.Printf("Finish 尾部样本: %.6f\n", wholeTail)
	fmt.Printf("统计: %+v\n", chunked.Stats())

	// ---------- 实际触发的失败:非法块整体拒绝,状态不变 ----------
	fmt.Println("\n=== 实际触发的失败 ===")
	bad, err := resampler.New(L, M, h)
	must(err)
	if _, err := bad.Feed([]float64{0.1, 0.2, 0.3}); err != nil {
		must(err)
	}
	before := bad.Stats()
	_, feedErr := bad.Feed([]float64{0.4, 1.5, -0.4}) // |1.5|>1,非法
	fmt.Printf("Feed 含 |x|>1 的块 -> 错误: %v\n", feedErr)
	after := bad.Stats()
	fmt.Printf("拒绝前后统计一致: %v (前 %+v,后 %+v)\n", before == after, before, after)

	// 状态未受影响,合法块继续处理。
	if _, err := bad.Feed([]float64{0.4, -0.4}); err != nil {
		must(err)
	}
	if _, err := bad.Finish(); err != nil {
		must(err)
	}
	_, afterFinishErr := bad.Feed([]float64{0.1})
	fmt.Printf("Finish 后再 Feed -> 错误: %v\n", afterFinishErr)
	if again, err := bad.Finish(); err == nil {
		fmt.Printf("重复 Finish -> 空成功(输出 %d 样本)\n", len(again))
	}

	fmt.Println("\n演示完成。")
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
