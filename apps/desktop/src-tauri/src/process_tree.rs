use std::io;

#[cfg(windows)]
mod platform {
    use super::io;
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::ptr::null;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    pub struct ProcessTree {
        job: HANDLE,
    }

    // Windows kernel handles may be moved between threads. This type never exposes the raw handle,
    // and all operations either use an immutable handle value or consume the owning wrapper.
    unsafe impl Send for ProcessTree {}
    unsafe impl Sync for ProcessTree {}

    impl ProcessTree {
        pub fn attach(pid: u32) -> io::Result<Self> {
            unsafe {
                let job = CreateJobObjectW(null(), null());
                if job.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) == 0
                {
                    let error = io::Error::last_os_error();
                    CloseHandle(job);
                    return Err(error);
                }
                let process = OpenProcess(
                    PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
                    0,
                    pid,
                );
                if process.is_null() {
                    let error = io::Error::last_os_error();
                    CloseHandle(job);
                    return Err(error);
                }
                let assigned = AssignProcessToJobObject(job, process);
                let error = if assigned == 0 {
                    Some(io::Error::last_os_error())
                } else {
                    None
                };
                CloseHandle(process);
                if let Some(error) = error {
                    CloseHandle(job);
                    return Err(error);
                }
                Ok(Self { job })
            }
        }

        pub fn terminate(&self) -> io::Result<()> {
            if unsafe { TerminateJobObject(self.job, 1) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
    }

    impl Drop for ProcessTree {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.job);
            }
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use super::io;

    pub struct ProcessTree;

    impl ProcessTree {
        pub fn attach(_pid: u32) -> io::Result<Self> {
            Ok(Self)
        }

        pub fn terminate(&self) -> io::Result<()> {
            Ok(())
        }
    }
}

pub use platform::ProcessTree;
