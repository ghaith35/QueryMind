import * as d3 from 'd3'
import { useEffect, useRef } from 'react'
import type { GraphNode, GraphEdge } from '../types/api'

const edgeKey = (edge: Pick<GraphEdge, 'source' | 'target'>): string =>
  `${edge.source}->${edge.target}`

const nodeColorByType = (type: string): string =>
  ({
    person: '#c084fc',
    organization: '#60a5fa',
    location: '#34d399',
    concept: '#fbbf24',
    technique: '#f87171',
  }[type] ?? '#94a3b8')

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  activeNodeIds: Set<string>
  activeEdgeIds: Set<string>
  onNodeClick?: (node: GraphNode) => void
}

export function KnowledgeGraph({ nodes, edges, activeNodeIds, activeEdgeIds, onNodeClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const simRef = useRef<d3.Simulation<any, any> | null>(null)
  const gRef = useRef<SVGGElement | null>(null)

  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const root = svg.append('g')
    gRef.current = root.node()

    root.append('g').attr('class', 'edges')
    root.append('g').attr('class', 'nodes')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.15, 6])
      .on('zoom', (event) => {
        root.attr('transform', event.transform)
      })

    svg.call(zoom)

    simRef.current = d3.forceSimulation<any>()
      .force('link', d3.forceLink<any, any>().id((d: any) => d.id).distance(82))
      .force('charge', d3.forceManyBody().strength(-210))
      .force('center', d3.forceCenter(0, 0))
      .force('collision', d3.forceCollide().radius((d: any) => 10 + Math.min(d.frequency ?? 1, 10) * 2))
      .alphaDecay(0.02)
      .velocityDecay(0.4)

    return () => {
      simRef.current?.stop()
      simRef.current = null
      gRef.current = null
      svg.on('.zoom', null)
      svg.selectAll('*').remove()
    }
  }, [])

  useEffect(() => {
    if (!containerRef.current || !svgRef.current || !simRef.current) return

    const updateSize = () => {
      if (!containerRef.current || !svgRef.current || !simRef.current) return
      const width = containerRef.current.clientWidth || 800
      const height = containerRef.current.clientHeight || 600
      d3.select(svgRef.current)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
      simRef.current.force('center', d3.forceCenter(width / 2, height / 2))
      if (nodes.length > 0) simRef.current.alpha(0.18).restart()
    }

    updateSize()
    const observer = new ResizeObserver(updateSize)
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [nodes.length])

  useEffect(() => {
    const sim = simRef.current
    const rootNode = gRef.current
    if (!sim || !rootNode) return

    const root = d3.select(rootNode)
    const hasActivePath = activeNodeIds.size > 0 || activeEdgeIds.size > 0

    const edgeSelection = root.select<SVGGElement>('g.edges')
      .selectAll<SVGLineElement, GraphEdge>('line')
      .data(edges, edgeKey)

    edgeSelection.exit()
      .transition()
      .duration(200)
      .attr('opacity', 0)
      .remove()

    const edgeEnter = edgeSelection.enter()
      .append('line')
      .attr('stroke', 'var(--edge-color)')
      .attr('stroke-width', (d) => 0.5 + d.weight * 2)
      .attr('opacity', 0)

    const allEdges = edgeEnter.merge(edgeSelection)
      .classed('edge-active', (d) => {
        const sourceId = (d.source as any).id ?? d.source
        const targetId = (d.target as any).id ?? d.target
        return activeEdgeIds.has(`${sourceId}->${targetId}`)
      })
      .attr('stroke', (d) => {
        const sourceId = (d.source as any).id ?? d.source
        const targetId = (d.target as any).id ?? d.target
        return activeEdgeIds.has(`${sourceId}->${targetId}`) ? 'var(--accent-glow)' : 'var(--edge-color)'
      })
      .attr('stroke-width', (d) => {
        const sourceId = (d.source as any).id ?? d.source
        const targetId = (d.target as any).id ?? d.target
        return activeEdgeIds.has(`${sourceId}->${targetId}`) ? 2.5 : 0.5 + d.weight * 2
      })
      .attr('opacity', (d) => {
        const sourceId = (d.source as any).id ?? d.source
        const targetId = (d.target as any).id ?? d.target
        const isActive = activeEdgeIds.has(`${sourceId}->${targetId}`)
        if (!hasActivePath) return 0.25 + d.weight * 0.5
        return isActive ? 1 : 0.1
      })

    edgeEnter.transition().duration(350).attr('opacity', (d) => 0.25 + d.weight * 0.5)

    const nodeSelection = root.select<SVGGElement>('g.nodes')
      .selectAll<SVGGElement, GraphNode>('g.node')
      .data(nodes, (d) => d.id)

    nodeSelection.exit()
      .transition()
      .duration(200)
      .attr('opacity', 0)
      .remove()

    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'node')
      .attr('opacity', 0)
      .style('cursor', 'pointer')
      .call(
        d3.drag<SVGGElement, any>()
          .on('start', (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart()
            d.fx = d.x
            d.fy = d.y
          })
          .on('drag', (event, d) => {
            d.fx = event.x
            d.fy = event.y
          })
          .on('end', (event, d) => {
            if (!event.active) sim.alphaTarget(0)
            d.fx = null
            d.fy = null
          })
      )
      .on('click', (_, d) => onNodeClick?.(d))

    nodeEnter.append('circle')
      .attr('r', (d) => 6 + Math.min(d.frequency, 10) * 1.5)
      .attr('fill', (d) => nodeColorByType(d.type))
      .attr('stroke', '#0f1117')
      .attr('stroke-width', 1.5)

    nodeEnter.append('text')
      .text((d) => d.label)
      .attr('dy', (d) => -(10 + Math.min(d.frequency, 10) * 1.5))
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-2)')
      .attr('font-size', '11px')
      .attr('font-family', 'var(--font-mono)')
      .attr('pointer-events', 'none')

    nodeEnter.append('title')
      .text((d) => `${d.label}\n${d.frequency} chunk${d.frequency > 1 ? 's' : ''}\n${d.document_sources.length} document${d.document_sources.length > 1 ? 's' : ''}`)

    const allNodes = nodeEnter.merge(nodeSelection)
      .classed('node-active', (d) => activeNodeIds.has(d.id))
      .attr('opacity', (d) => (!hasActivePath || activeNodeIds.has(d.id) ? 1 : 0.38))

    allNodes.select('circle')
      .attr('r', (d) => 6 + Math.min(d.frequency, 10) * 1.5)
      .attr('fill', (d) => nodeColorByType(d.type))

    allNodes.select('text')
      .attr('fill', (d) => (!hasActivePath || activeNodeIds.has(d.id) ? 'var(--text-2)' : 'var(--text-muted)'))

    nodeEnter.transition().duration(350).attr('opacity', 1)

    sim.nodes(nodes as any)
    ;(sim.force('link') as d3.ForceLink<any, any>).links(edges as any)

    if (nodes.length === 0) {
      sim.stop()
      return
    }

    sim.alpha(0.3).restart()
    sim.on('tick', () => {
      allEdges
        .attr('x1', (d: any) => d.source.x ?? 0)
        .attr('y1', (d: any) => d.source.y ?? 0)
        .attr('x2', (d: any) => d.target.x ?? 0)
        .attr('y2', (d: any) => d.target.y ?? 0)

      allNodes.attr('transform', (d: any) => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })
  }, [nodes, edges, activeNodeIds, activeEdgeIds, onNodeClick])

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <svg ref={svgRef} className="h-full w-full" />
      {nodes.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-6">
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-1)]/92 px-5 py-4 text-center shadow-[var(--shadow-sm)] backdrop-blur">
            <div className="text-sm font-mono text-[var(--text-1)]">Graph is waiting for data</div>
            <div className="mt-2 max-w-sm text-sm leading-relaxed text-[var(--text-2)]">
              Upload documents or loosen the filters. If the layout ever feels stuck, use the rebuild button to start the graph from a clean canvas.
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
