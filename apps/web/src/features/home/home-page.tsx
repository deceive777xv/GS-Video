import { useState } from 'react'

import type { ProjectSummaryDto } from '../../api/types'

const STEP_LABELS: Record<ProjectSummaryDto['workflow_step'], string> = {
  import: '导入素材',
  subject: '选择人物',
  camera: '调整机位',
  preview: '预览合成',
  postprocess: '合成后处理',
  export: '导出成片',
}

interface HomePageProps {
  activeProjectId: string | null
  busy: boolean
  projects: ProjectSummaryDto[]
  onCreate(name: string): Promise<void>
  onDelete(project: ProjectSummaryDto): Promise<void>
  onOpen(project: ProjectSummaryDto): Promise<void>
  onRename(project: ProjectSummaryDto, name: string): Promise<void>
}

export function HomePage({
  activeProjectId,
  busy,
  projects,
  onCreate,
  onDelete,
  onOpen,
  onRename,
}: HomePageProps) {
  const [name, setName] = useState('')
  const [editing, setEditing] = useState<ProjectSummaryDto | null>(null)
  const [deleting, setDeleting] = useState<ProjectSummaryDto | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const create = async (): Promise<void> => {
    if (name.trim() === '' || submitting) return
    setSubmitting(true)
    try {
      await onCreate(name)
      setName('')
    } catch {
      // The app shell owns the visible error banner; preserve the entered name.
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="hub-main">
      <section className="hub-hero">
        <div>
          <p className="eyebrow">PROJECT HOME</p>
          <h1>从一个项目继续，或开始新的合成。</h1>
          <p>项目库保存配置与共享素材；帧、遮罩和生成结果统一进入可迁移的缓存目录。</p>
        </div>
        <form className="project-create" onSubmit={(event) => { event.preventDefault(); void create() }}>
          <label htmlFor="new-project-name">新项目名称</label>
          <div>
            <input id="new-project-name" maxLength={80} onChange={(event) => setName(event.target.value)} placeholder="例如：城市夜景人物合成" value={name} />
            <button disabled={busy || submitting || name.trim() === ''} type="submit">创建项目</button>
          </div>
          {busy ? <small>当前任务运行中，完成或取消后才能创建、切换或删除项目。</small> : null}
        </form>
      </section>

      <section className="project-section" aria-labelledby="project-list-title">
        <div className="section-heading">
          <div><p className="eyebrow">PROJECTS</p><h2 id="project-list-title">项目</h2></div>
          <span>{projects.length} 个项目</span>
        </div>
        {projects.length === 0 ? (
          <div className="empty-state"><strong>还没有项目</strong><p>在上方输入名称创建第一个项目。</p></div>
        ) : (
          <div className="project-grid">
            {projects.map((item) => (
              <article className={`project-card ${item.project_id === activeProjectId ? 'is-active' : ''}`} key={item.project_id}>
                <div className="project-card-top">
                  <span>{item.project_id === activeProjectId ? '当前项目' : '本地项目'}</span>
                  <button
                    aria-label={`删除项目 ${item.name}`}
                    className="project-icon-button project-delete-button"
                    disabled={busy}
                    onClick={() => setDeleting(item)}
                    title="删除项目"
                    type="button"
                  >×</button>
                </div>
                <div className="project-title-row">
                  <h3>{item.name}</h3>
                  <button
                    aria-label={`重命名项目 ${item.name}`}
                    className="project-icon-button project-rename-button"
                    onClick={() => { setEditing(item); setName(item.name) }}
                    title="重命名项目"
                    type="button"
                  >✎</button>
                </div>
                <dl>
                  <div><dt>最近更新</dt><dd>{new Date(item.updated_at).toLocaleString()}</dd></div>
                  <div><dt>流程进度</dt><dd>{STEP_LABELS[item.workflow_step]}</dd></div>
                </dl>
                <div className="project-actions">
                  <button disabled={busy} onClick={() => { void onOpen(item).catch(() => undefined) }} type="button">继续制作</button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      {editing !== null ? (
        <div className="modal-backdrop" role="presentation">
          <form className="modal-card" onSubmit={(event) => {
            event.preventDefault()
            if (name.trim() === '') return
            setSubmitting(true)
            void onRename(editing, name).then(() => { setEditing(null); setName('') }).catch(() => undefined).finally(() => setSubmitting(false))
          }}>
            <p className="eyebrow">RENAME</p><h2>重命名项目</h2>
            <input autoFocus maxLength={80} onChange={(event) => setName(event.target.value)} value={name} />
            <div className="modal-actions"><button className="button-secondary" onClick={() => { setEditing(null); setName('') }} type="button">取消</button><button disabled={submitting} type="submit">保存</button></div>
          </form>
        </div>
      ) : null}

      {deleting !== null ? (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-card" role="alertdialog" aria-labelledby="delete-title">
            <p className="eyebrow">DELETE PROJECT</p><h2 id="delete-title">删除“{deleting.name}”？</h2>
            <p>项目配置、缓存和生成结果会被删除。共享素材库中的视频和 PLY 不受影响。</p>
            <div className="modal-actions"><button className="button-secondary" onClick={() => setDeleting(null)} type="button">取消</button><button className="button-danger" disabled={submitting} onClick={() => {
              setSubmitting(true)
              void onDelete(deleting).then(() => setDeleting(null)).catch(() => undefined).finally(() => setSubmitting(false))
            }} type="button">确认删除</button></div>
          </div>
        </div>
      ) : null}
    </main>
  )
}
