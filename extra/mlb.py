from tinygrad.multi import MultiLazyBuffer
def print_lb(lb):
  def repr_lb(lb, indent=0):
    def _indent(): return " " * indent
    text = ""
    if isinstance(lb, MultiLazyBuffer):
      text += _indent() + "MLB "
      if lb.axis is not None:
        text += _indent() + f"shape {lb.shape} axis {lb.axis}"
        text += _indent() + f" bounds {lb.bounds}"
      text += "\n"
      indent += 4
      text += _indent()
      for i, _lb in enumerate(lb.lbs):
        text += repr_lb(_lb, indent)
        if i < len(lb.lbs) - 1: text += _indent()
    else:
      text += "LB"
      view = lb.st.views[0]
      text += f" shape {view.shape}"
      text += f" strides {view.strides}"
      text += f" offset {view.offset}" if view.offset != 0 else ""
      text += f" mask {view.mask}" if view.mask is not None else ""
      text += "\n"
      indent += 3
      if hasattr(lb, "_base") and lb._base is not None:
        text += _indent() + "_base: "
        text += repr_lb(lb._base, indent + 7)
        return text
      else:
        text += _indent() + f"{lb.op} "
        text += f"arg: {lb.arg} " if lb.arg is not None else ""
        text += f"size: {lb.size} "
        text += f"device: {lb.device} \n"
        if hasattr(lb, "buffer") and hasattr(lb.buffer, "_buf"):
          return text
        elif len(lb.srcs) > 0:
          text += _indent() + "srcs: "
          indent += 6
          for i, src in enumerate(lb.srcs):
            text += repr_lb(src, indent)
            if i < len(lb.srcs) - 1: text += _indent()
    return text
  print(repr_lb(lb))
