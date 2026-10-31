
type ButtonProps = {
    buttonText: string;
    onClick: () => void;
};

export function Button({buttonText, onClick, ...props}: ButtonProps) {
    return <button type="button" onClick={onClick} {...props}>
        {buttonText}
    </button>
}