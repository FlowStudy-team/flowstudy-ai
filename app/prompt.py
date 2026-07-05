from .schemas import AiContext


def build_system_prompt(context: AiContext) -> str:
    parts = [
        "You are FlowStudy AI, a learning assistant integrated into a programming education platform. "
        "Your role is to help users understand tutorials, debug code submissions, and learn algorithms.\n",
        "Guidelines:\n"
        "- Be concise and focused on the user's immediate question.\n"
        "- When reviewing code errors, explain the logical root cause rather than just providing the correct answer.\n"
        "- Reference specific details from the context below when relevant.\n"
        "- If the user asks about code, focus on logic, edge cases, and performance, referencing the provided code.\n"
        "- For compilation errors, explain the error message in simple terms.\n",
    ]

    ctx_lines = []

    read_ctx = []
    if context.tutorialTitle:
        read_ctx.append(f"- Tutorial: {context.tutorialTitle}")
    if context.blogTitle:
        read_ctx.append(f"- Blog article: {context.blogTitle}")
    if read_ctx:
        ctx_lines.append("READING CONTEXT:")
        ctx_lines.extend(read_ctx)
        ctx_lines.append("")

    prob_ctx = []
    if context.problemTitle:
        prob_ctx.append(f"- Problem: {context.problemTitle}")
    if context.problemDescription:
        desc = context.problemDescription[:2000]
        prob_ctx.append(f"- Problem description: {desc}")
    if context.language:
        prob_ctx.append(f"- Language: {context.language}")
    if context.userCode:
        code = context.userCode[:3000]
        prob_ctx.append(
            f"- User's code:\n```{context.language or ''}\n{code}\n```"
        )
    if prob_ctx:
        ctx_lines.append("PROBLEM CONTEXT:")
        ctx_lines.extend(prob_ctx)
        ctx_lines.append("")

    sub_ctx = []
    if context.submissionStatus:
        sub_ctx.append(f"- Submission status: {context.submissionStatus}")
    if context.failedCaseInput:
        sub_ctx.append(f"- Failed test case input: {context.failedCaseInput}")
    if context.expectedOutput:
        sub_ctx.append(f"- Expected output: {context.expectedOutput}")
    if context.actualOutput:
        sub_ctx.append(f"- Actual output: {context.actualOutput}")
    if context.compileMessage:
        sub_ctx.append(f"- Compile error: {context.compileMessage}")
    if sub_ctx:
        ctx_lines.append("SUBMISSION RESULT:")
        ctx_lines.extend(sub_ctx)

    if ctx_lines:
        parts.append("\nCURRENT CONTEXT:")
        parts.extend(ctx_lines)

    return "\n".join(parts)
