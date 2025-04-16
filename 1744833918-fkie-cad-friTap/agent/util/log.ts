export function log(str: string) {
    var message: { [key: string]: string } = {}
    message["contentType"] = "console"
    message["console"] = str
    send(message)
}


export function devlog(str: string) {
    var message: { [key: string]: string } = {}
    message["contentType"] = "console_dev"
    message["console_dev"] = str
    send(message)
}


export function devlog_error(str: string) {
    var message: { [key: string]: string } = {}
    message["contentType"] = "console_error"
    message["console_error"] = str
    send(message)
}